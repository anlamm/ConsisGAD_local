
## copied from baseline.py

import argparse
import sys
import os
import csv
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from modules.data_loader import get_index_loader_test, load_ego_graphs, setup_training_dataloder, setup_eval_dataloder
from models import simpleGNN_MR
# from models_edges import simpleGNN_MR
# from baseline_model import simpleGNN_MR   ###### Modify to formal GCN
import modules.mod_utls as m_utls
from modules.loss import nll_loss, l2_regularization, nll_loss_raw
from modules.evaluation import eval_pred
from modules.aux_mod import fixed_augmentation
from sklearn.metrics import f1_score
from modules.conv_mod import CustomLinear
from modules.mr_conv_mod import build_mlp
import numpy as np
from numpy import random
import math
import pandas as pd
from functools import partial
import dgl
import warnings
import wandb
import yaml
warnings.filterwarnings("ignore")


from modules.utils import save_results
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict


class SoftAttentionDrop(nn.Module):
    def __init__(self, args):
        super(SoftAttentionDrop, self).__init__()
        dim = args['hidden-dim']
        
        self.temp = args['trainable-temp']
        self.p = args['trainable-drop-rate']
        if args['trainable-model'] == 'proj':
            self.mask_proj = CustomLinear(dim, dim)
        else:
            self.mask_proj = build_mlp(in_dim=dim, out_dim=dim, p=args['mlp-drop'], final_act=False)
        
        self.detach_y = args['trainable-detach-y']
        self.div_eps = args['trainable-div-eps']
        self.detach_mask = args['trainable-detach-mask']
        
    def forward(self, feature, in_eval=False):
        mask = self.mask_proj(feature)

        y = torch.zeros_like(mask)
        k = round(mask.shape[1] * self.p)

        for _ in range(k):
            if self.detach_y:
                w = torch.zeros_like(y)
                w[y>0.5] = 1
                w = (1. - w).detach()
            else:
                w = (1. - y)
                
            logw = torch.log(w + 1e-12)
            y1 = (mask + logw) / self.temp
            y1 = y1 - torch.amax(y1, dim=1, keepdim=True)
            
            if self.div_eps:
                y1 = torch.exp(y1) / (torch.sum(torch.exp(y1), dim=1, keepdim=True) + args['trainable-eps'])
            else:
                y1 = torch.exp(y1) / torch.sum(torch.exp(y1), dim=1, keepdim=True)
                
            y = y + y1 * w
            
        mask = 1. - y
        mask = mask / (1. - self.p)
        
        if in_eval and self.detach_mask:
            mask = mask.detach()
            
        return feature * mask


def create_model(args, e_ts):
    if args['model'] == 'backbone':
        tmp_model = simpleGNN_MR(in_feats=args['node-in-dim'], hidden_feats=args['hidden-dim'],
                                 out_feats=args['node-out-dim'], 
                                 num_layers=args['num-layers'], e_types=e_ts, input_drop=args['input-drop'], hidden_drop=args['hidden-drop'], 
                                 mlp_drop=args['mlp-drop'], mlp12_dim=args['mlp12-dim'], mlp3_dim=args['mlp3-dim'], bn_type=args['bn-type'])
    else:
        raise
    tmp_model.to(args['device'])
            
    return tmp_model


def UDA_train_epoch(epoch, model, loss_func, graph, label_loader, unlabel_loader, optimizer, augmentor, args):
    model.train()
    num_iters = args['train-iterations']
    
    sampler, attn_drop, ad_optim = augmentor
    
    unlabel_loader_iter = iter(unlabel_loader)
    label_loader_iter = iter(label_loader)
    
    for idx in range(num_iters):
        try:
            label_idx = label_loader_iter.__next__()
        except:
            label_loader_iter = iter(label_loader)
            label_idx = label_loader_iter.__next__()
        try:
            unlabel_idx = unlabel_loader_iter.__next__()
        except:
            unlabel_loader_iter = iter(unlabel_loader)
            unlabel_idx = unlabel_loader_iter.__next__()

        if epoch > args['trainable-warm-up']:
            model.eval()
            with torch.no_grad():
                _, _, u_blocks = fixed_augmentation(graph, unlabel_idx.to(args['device']), sampler, aug_type='none')
                weak_inter_results = model(u_blocks, update_bn=False, return_logits=True)
                weak_h = torch.stack(weak_inter_results, dim=1)
                weak_h = weak_h.reshape(weak_h.shape[0], -1)
                weak_logits = model.proj_out(weak_h)
                u_pred_weak_log = weak_logits.log_softmax(dim=-1)
                u_pred_weak = u_pred_weak_log.exp()[:, 1]
                
            pseudo_labels = torch.ones_like(u_pred_weak).long()
            neg_tar = (u_pred_weak <= (args['normal-th']/100.)).bool()
            pos_tar = (u_pred_weak >= (args['fraud-th']/100.)).bool()
            pseudo_labels[neg_tar] = 0
            pseudo_labels[pos_tar] = 1
            u_mask = torch.logical_or(neg_tar, pos_tar)

            model.train()
            attn_drop.train()
            for param in model.parameters():
                param.requires_grad = False
            for param in attn_drop.parameters():
                param.requires_grad = True

            _, _, u_blocks = fixed_augmentation(graph, unlabel_idx.to(args['device']), sampler, aug_type='drophidden')

            inter_results = model(u_blocks, update_bn=False, return_logits=True)
            dropped_results = [inter_results[0]]
            for i in range(1, len(inter_results)):
                dropped_results.append(attn_drop(inter_results[i]))
            h = torch.stack(dropped_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            logits = model.proj_out(h)
            u_pred = logits.log_softmax(dim=-1)
            
            consistency_loss = nll_loss_raw(u_pred, pseudo_labels, pos_w=1.0, reduction='none')
            consistency_loss = torch.mean(consistency_loss * u_mask)

            if args['diversity-type'] == 'cos':
                diversity_loss = F.cosine_similarity(weak_h, h, dim=-1)
            elif args['diversity-type'] == 'euc':
                diversity_loss = F.pairwise_distance(weak_h, h)
            else:
                raise
            diversity_loss = torch.mean(diversity_loss * u_mask)
            
            total_loss = args['trainable-consis-weight'] * consistency_loss - diversity_loss + args['trainable-weight-decay'] * l2_regularization(attn_drop)
            
            ad_optim.zero_grad()
            total_loss.backward()
            ad_optim.step()
            
            for param in model.parameters():
                param.requires_grad = True
            for param in attn_drop.parameters():
                param.requires_grad = False

            inter_results = model(u_blocks, update_bn=False, return_logits=True)
            dropped_results = [inter_results[0]]
            for i in range(1, len(inter_results)):
                dropped_results.append(attn_drop(inter_results[i], in_eval=True))

            h = torch.stack(dropped_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            logits = model.proj_out(h)
            u_pred = logits.log_softmax(dim=-1)

            unsup_loss = nll_loss_raw(u_pred, pseudo_labels, pos_w=1.0, reduction='none')
            unsup_loss = torch.mean(unsup_loss * u_mask)
        else:
            unsup_loss = 0.0

        _, _, s_blocks = fixed_augmentation(graph, label_idx.to(args['device']), sampler, aug_type='none')
        s_pred = model(s_blocks)
        s_target = s_blocks[-1].dstdata['label']
            
        sup_loss, _ = loss_func(s_pred, s_target)

        loss = sup_loss + unsup_loss + args['weight-decay'] * l2_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     

        if idx % 10 == 0:
            print(f"Iter {idx+1}/{num_iters}, loss: {loss.item()}")
        

def simple_train_epoch(epoch, model, loss_func, graph, label_loader, unlabel_loader, optimizer, augmentor, args):
    model.train()
    num_iters = args['train-iterations']
    
    sampler, attn_drop, ad_optim = augmentor
    
    unlabel_loader_iter = iter(unlabel_loader)
    label_loader_iter = iter(label_loader)
    
    losses = []
    for idx in range(num_iters):
        try:
            label_idx = label_loader_iter.__next__()
        except:
            label_loader_iter = iter(label_loader)
            label_idx = label_loader_iter.__next__()
        try:
            unlabel_idx = unlabel_loader_iter.__next__()
        except:
            unlabel_loader_iter = iter(unlabel_loader)
            unlabel_idx = unlabel_loader_iter.__next__()

        _, _, s_blocks = fixed_augmentation(graph, label_idx.to(args['device']), sampler, aug_type='none')
        s_pred = model(s_blocks)
        s_target = s_blocks[-1].dstdata['label']
            
        sup_loss, _ = loss_func(s_pred, s_target)

        # loss = sup_loss + unsup_loss + args['weight-decay'] * l2_regularization(model)
        loss = sup_loss + args['weight-decay'] * l2_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     

        # if idx % 10 == 0:
        #     print(f"Iter {idx+1}/{num_iters}, loss: {loss.item()}")

        losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{args['epochs']}, loss: {np.mean(losses)}")


def subgraph_classification_train_epoch(epoch, model, loss_func, graph, label_loader, unlabel_loader, optimizer, augmentor, args):

    ### model training
    model.train()
    num_iters = args['train-iterations']
    
    sampler, attn_drop, ad_optim = augmentor
    label_loader_iter = iter(label_loader)
    
    losses = []
    for idx in range(num_iters):
        try:
            sg, label_idx = label_loader_iter.__next__()
        except:
            label_loader_iter = iter(label_loader)
            sg, label_idx = label_loader_iter.__next__()

        # s_blocks = [dgl.to_block(sg.to(args['device']), label_idx.to(args['device']))]
        _, _, s_blocks = fixed_augmentation(sg.to(args['device']), label_idx.to(args['device']), sampler, aug_type='none')
        s_pred = model(s_blocks)
        s_target = s_blocks[-1].dstdata['label']
            
        sup_loss, _ = loss_func(s_pred, s_target)

        # loss = sup_loss + unsup_loss + args['weight-decay'] * l2_regularization(model)
        loss = sup_loss + args['weight-decay'] * l2_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     

        # if idx % 10 == 0:
        #     print(f"Iter {idx+1}/{num_iters}, loss: {loss.item()}")

        losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{args['epochs']}, loss: {np.mean(losses)}")


def get_model_pred(model, graph, data_loader, sampler, args, return_logits=False):
    model.eval()
    
    pred_list = []
    target_list = []
    emb_list = []
    with torch.no_grad():
        for node_idx in data_loader:
            if args['ego']:
                sg, node_idx = node_idx
                # blocks = [dgl.to_block(sg, node_idx.to(args['device']))]
                _, _, blocks = sampler.sample_blocks(sg.to(args['device']), node_idx.to(args['device']))
            else:
                _, _, blocks = sampler.sample_blocks(graph, node_idx.to(args['device']))
            
            
            if return_logits:
                emb = model(blocks, return_logits=True)
                emb = torch.stack(emb, dim=1)
                emb = emb.reshape(emb.shape[0], -1)
                emb_list.append(emb)

            pred = model(blocks)  
            target = blocks[-1].dstdata['label']
            
            pred_list.append(pred.detach())
            target_list.append(target.detach())
        pred_list = torch.cat(pred_list, dim=0)
        target_list = torch.cat(target_list, dim=0)
        pred_list = pred_list.exp()[:, 1]
        
        if return_logits:
            emb_list = torch.cat(emb_list, dim=0)
            return pred_list, target_list, emb_list
    return pred_list, target_list


def val_epoch(epoch, model, graph, valid_loader, test_loader, sampler, args):
    valid_dict = {}
    valid_pred, valid_target = get_model_pred(model, graph, valid_loader, sampler, args)
    v_roc, v_pr, _, _, _, _, v_f1, v_thre = eval_pred(valid_pred, valid_target)
    valid_dict['auc-roc'] = v_roc
    valid_dict['auc-pr'] = v_pr
    valid_dict['binary f1'] = v_f1
        
    test_dict = {}
    test_pred, test_target = get_model_pred(model, graph, test_loader, sampler, args)
    t_roc, t_pr, _, _, _, _, _, _ = eval_pred(test_pred, test_target)
    test_dict['auc-roc'] = t_roc
    test_dict['auc-pr'] = t_pr
    
    test_pred = test_pred.cpu().numpy()
    test_target = test_target.cpu().numpy()
    guessed_target = np.zeros_like(test_target)
    guessed_target[test_pred > v_thre] = 1
    t_f1 = f1_score(test_target, guessed_target, average='binary')
    test_dict['binary f1'] = t_f1
            
    return valid_dict, test_dict



def run_model(args):
    graph, label_loader, valid_loader, test_loader, unlabel_loader = get_index_loader_test(name=args['data-set'], 
                                                                                           batch_size=args['batch-size'], 
                                                                                           unlabel_ratio=args['unlabel-ratio'],
                                                                                           training_ratio=args['training-ratio'],
                                                                                           shuffle_train=args['shuffle-train'], 
                                                                                           to_homo=args['to-homo'],
                                                                                           random_feature=args['random_feature'],
                                                                                           structural_feature=args['structural_feature'],
                                                                                           same_feature=args['same_feature'],
                                                                                           cat_feature=args['cat_feature'],
                                                                                           verbose=args['debug'],
                                                                                           load_offline=True,
                                                                                           seed = args['seed'],
                                                                                           add_edge_feature=args['add_edge_feature'])
    
    if args['ego']:
        graph = graph.to(args['device'])

        ego_nodes_train, ego_nodes_val, ego_nodes_test = load_ego_graphs(name=args['data-set'], seed=args['seed'], size=256)
        label_loader = setup_training_dataloder(
            'lc', ego_nodes_train, graph, graph.ndata['feature'], batch_size=args['batch-size'])
        valid_loader = setup_eval_dataloder(
            'lc', graph, graph.ndata['feature'], ego_nodes_val, batch_size=args['batch-size'])
        test_loader = setup_eval_dataloder(
            'lc', graph, graph.ndata['feature'], ego_nodes_test, batch_size=args['batch-size'])

    

    if args['drop_edges']:
        for etype in graph.etypes:
            nedges = graph.num_edges(etype=etype)
            graph.remove_edges(torch.arange(nedges), etype=etype)
        print(f"#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}")
    
    graph = graph.to(args['device'])
    print(f"#Features: {graph.ndata['feature'].shape}")

    if args['debug']:
        exit(0)
    
    args['node-in-dim'] = graph.ndata['feature'].shape[1]
    args['node-out-dim'] = 2
    if args['add_edge_feature']:
        for et in graph.etypes:
            args['edge-in-dim'] = graph.edges[et].data['eh'].shape[1]
    
    my_model = create_model(args, graph.etypes)
    print(f"#Params: {num_params(my_model)}")
    
    if args['optim'] == 'adam':
        optimizer = optim.Adam(my_model.parameters(), lr=args['lr'], weight_decay=0.0)
    elif args['optim'] == 'rmsprop':
        optimizer = optim.RMSprop(my_model.parameters(), lr=args['lr'], weight_decay=0.0)
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num-layers'])
    
    # train_epoch = UDA_train_epoch
    if args['ego']:
        train_epoch = subgraph_classification_train_epoch
    else:
        train_epoch = simple_train_epoch
    attn_drop = SoftAttentionDrop(args).to(args['device'])
    if args['trainable-optim'] == 'rmsprop':
        ad_optim = optim.RMSprop(attn_drop.parameters(), lr=args['trainable-lr'], weight_decay=0.0)
    else:
        ad_optim = optim.Adam(attn_drop.parameters(), lr=args['trainable-lr'], weight_decay=0.0)
    augmentor = (sampler, attn_drop, ad_optim)

    task_loss = nll_loss
    
    best_val = sys.float_info.min
    for epoch in range(args['epochs']):
        train_epoch(epoch, my_model, task_loss, graph, label_loader, unlabel_loader, optimizer, augmentor, args)
        val_results, test_results = val_epoch(epoch, my_model, graph, valid_loader, test_loader, sampler, args)
        
        if val_results['auc-roc'] > best_val:
            print(f"Current Best Epoch: {epoch+1}")

            best_val = val_results['auc-roc']
            test_in_best_val = test_results
            
            if args['store-model']:
                m_utls.store_model(my_model, args)
                
    return list(test_in_best_val.values())

def load_model(args, ets):
    file_path = os.path.join(f"{args['save_path']}",
                             f"{args['data-set']}-{args['seed']}.pth")
    model = create_model(args, ets)
    model.load_state_dict(torch.load(file_path))

    return model

def run_inference(args):
    graph, label_loader, valid_loader, test_loader, unlabel_loader = get_index_loader_test(name=args['data-set'], 
                                                                                           batch_size=args['batch-size'], 
                                                                                           unlabel_ratio=args['unlabel-ratio'],
                                                                                           training_ratio=args['training-ratio'],
                                                                                           shuffle_train=args['shuffle-train'], 
                                                                                           to_homo=args['to-homo'],
                                                                                           random_feature=args['random_feature'],
                                                                                           structural_feature=args['structural_feature'],
                                                                                           same_feature=args['same_feature'],
                                                                                           cat_feature=args['cat_feature'],
                                                                                           verbose=args['debug'],
                                                                                           load_offline=True,
                                                                                           seed = args['seed'],
                                                                                           add_edge_feature=args['add_edge_feature'])

    graph = graph.to(args['device'])
    print(f"#Features: {graph.ndata['feature'].shape}")

    if args['debug']:
        exit(0)
    
    args['node-in-dim'] = graph.ndata['feature'].shape[1]
    args['node-out-dim'] = 2
    if args['add_edge_feature']:
        for et in graph.etypes:
            args['edge-in-dim'] = graph.edges[et].data['eh'].shape[1]
    
    # my_model = create_model(args, graph.etypes)
    my_model = load_model(args, graph.etypes)
    print(f"#Params: {num_params(my_model)}")
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num-layers'])
    attn_drop = SoftAttentionDrop(args).to(args['device'])
    if args['trainable-optim'] == 'rmsprop':
        ad_optim = optim.RMSprop(attn_drop.parameters(), lr=args['trainable-lr'], weight_decay=0.0)
    else:
        ad_optim = optim.Adam(attn_drop.parameters(), lr=args['trainable-lr'], weight_decay=0.0)
    augmentor = (sampler, attn_drop, ad_optim)

    my_model.eval()
    test_pred, test_target, test_emb = get_model_pred(my_model, graph, test_loader, sampler, args, return_logits=True)
    test_pred = test_pred.cpu()
    test_target = test_target.cpu()
    guessed_target = torch.zeros_like(test_target)
    guessed_target[test_pred > args['thres']] = 1

    print(f'shapes: {guessed_target.shape} {test_target.shape} {test_emb.shape}')

    nodeids = []
    for batch_idx in test_loader:
        for idx in batch_idx:
            nodeids.append(idx.data)
    nodeids = torch.LongTensor(nodeids)

    nodeids = nodeids[torch.nonzero(guessed_target == 1, as_tuple=True)[0]]
    subgraph = dgl.node_subgraph(graph, nodeids.to(graph.device), relabel_nodes=True)
    test_emb = test_emb[torch.nonzero(guessed_target == 1, as_tuple=True)[0]]


    from networkx.algorithms.community.label_propagation import label_propagation_communities
    from networkx.algorithms.community.louvain import louvain_communities
    from sklearn.cluster import KMeans, SpectralClustering as SC
    from clustering import _compute_leiden_communities, aggmmr_cluster
    from clustering.utils import to_df

    community_detection_algos = {
        'LP': label_propagation_communities,
        'Louvain': louvain_communities,
        'KMeans': KMeans,
        'Spectral': SC,
        'Hierarchical_Leiden': _compute_leiden_communities,
        'AGGMMR': aggmmr_cluster,
    }


    def community_detection_warper(algo_name, input_dglgraph) -> Dict:
        import networkx as nx

        input_dglgraph = input_dglgraph.to('cpu')
        input_dglgraph = dgl.to_homogeneous(input_dglgraph)
        input_nxgraph = dgl.to_networkx(input_dglgraph)
        input_nxgraph = nx.Graph(input_nxgraph)
        
        func = community_detection_algos.get(algo_name, None)
        if algo_name == 'LP':
            # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.label_propagation.label_propagation_communities.html#networkx.algorithms.community.label_propagation.label_propagation_communities
            ret = func(input_nxgraph)

            node2comm = {}
            for comm, nodeids in enumerate(list(ret)):
                for nodeid in nodeids:
                    node2comm[nodeid] = comm

            df = pd.DataFrame({'nodeIds': nodeids, 'Community ID': list(node2comm.values())}, columns = ['nodeIds', 'Community ID'])
        elif algo_name == 'Louvain':
            # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html#networkx.algorithms.community.louvain.louvain_communities
            ret = func(input_nxgraph)

            node2comm = {}
            for comm, nodeids in enumerate(ret):
                for nodeid in nodeids:
                    node2comm[nodeid] = comm
            
            df = pd.DataFrame({'nodeIds': nodeids, 'Community ID': list(node2comm.values())}, columns = ['nodeIds', 'Community ID'])
        elif algo_name == 'KMeans' or algo_name == 'Spectral':
            n_classes = 5
            clustering = func(n_clusters=n_classes)
            ret = clustering.fit_predict(test_emb.cpu().numpy())

            node2comm = {}
            for nodeid, comm in enumerate(ret):
                node2comm[nodeid] = comm
            
            df = pd.DataFrame({'nodeIds': nodeids, 'Community ID': list(node2comm.values())}, columns = ['nodeIds', 'Community ID'])
        elif algo_name == 'Hierarchical_Leiden':
            max_cluster_size = 20
            use_lcc=False

            node_id_to_community_map = func(
                input_nxgraph,
                max_cluster_size=max_cluster_size,
                use_lcc=use_lcc,
                seed=0xDEADBEEF,
            )

            df = to_df(node_id_to_community_map=node_id_to_community_map)
            df['avg_pred'] = df.apply(lambda r: test_pred.numpy()[np.array(r['nodeids'])].mean(), axis=1)
            df['max_pred'] = df.apply(lambda r: test_pred.numpy()[np.array(r['nodeids'])].max(), axis=1)
            df['min_pred'] = df.apply(lambda r: test_pred.numpy()[np.array(r['nodeids'])].min(), axis=1)
            df['pred@75'] = df.apply(lambda r: np.quantile(test_pred.numpy()[np.array(r['nodeids'])], .75), axis=1)
            df['pred@50'] = df.apply(lambda r: np.quantile(test_pred.numpy()[np.array(r['nodeids'])], .50), axis=1)
            df['pred@25'] = df.apply(lambda r: np.quantile(test_pred.numpy()[np.array(r['nodeids'])], .25), axis=1)
            
        elif algo_name == 'AGGMMR':
            nclass = 50
            features = test_emb.cpu().numpy()
            max_cluster_size = 20
            use_lcc=False

            node_id_to_community_map = func(
                input_nxgraph,
                features,
                nclass,
                max_cluster_size=max_cluster_size,
                use_lcc=use_lcc,
                seed=0xDEADBEEF,
            )

            df = to_df(node_id_to_community_map=node_id_to_community_map)
            df['avg_pred'] = df.apply(lambda r: test_pred.numpy()[np.array(r['nodeids'])].mean(), axis=1)
            df['max_pred'] = df.apply(lambda r: test_pred.numpy()[np.array(r['nodeids'])].max(), axis=1)
            df['min_pred'] = df.apply(lambda r: test_pred.numpy()[np.array(r['nodeids'])].min(), axis=1)
            df['pred@75'] = df.apply(lambda r: np.quantile(test_pred.numpy()[np.array(r['nodeids'])], .75), axis=1)
            df['pred@50'] = df.apply(lambda r: np.quantile(test_pred.numpy()[np.array(r['nodeids'])], .50), axis=1)
            df['pred@25'] = df.apply(lambda r: np.quantile(test_pred.numpy()[np.array(r['nodeids'])], .25), axis=1)
        else:
            raise NotImplementedError(f'Unknown community detection algorithm: {algo_name}')
        
        return df

    df = community_detection_warper(args['comm_algo'], subgraph)

    # print(len(nodeids), len(node2comm))
    # df = pd.DataFrame({'nodeIds': nodeids, 'Community ID': list(node2comm.values())}, columns = ['nodeIds', 'Community ID'])
    df.to_csv('black_communities.csv', index=False)
    print(f'Community results saved to ./black_communities.csv')

    return 0
    



def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def num_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params

if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs. Default is 1.')
    parser.add_argument('--random_feature', action="store_true")
    parser.add_argument('--structural_feature', action="store_true")
    parser.add_argument('--same_feature', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--save_path', type=str, default="model-weights", help="path for saving model weights")
    parser.add_argument('--drop_edges', action="store_true")
    parser.add_argument('--cat_feature', action="store_true")
    parser.add_argument('--ego', action="store_true")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--add_edge_feature', action="store_true")
    parser.add_argument('--num_layers', type=int, default=1)

    parser.add_argument('--thres', type=float, default=0.5, help=">thres are predicted as black users")
    parser.add_argument('--comm_algo', type=str, default='LP', choices=['LP', 'Louvain', 'KMeans', 'Spectral', 'Hierarchical_Leiden', 'AGGMMR'], help='Algorithm to detect communities')


    args0 = parser.parse_args()
    cfg = vars(parser.parse_args())
    
    args = get_config(cfg['config'])
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:%d'%(args['device']))
    else:
        args['device'] = torch.device('cpu')
    args['device'] = args0.device
    
    # args['epochs'] = 300
    args['debug'] = False
    if args0.debug:
        args['debug'] = True
    args['save_path'] = args0.save_path
    args['cat_feature'] = False
    if args0.cat_feature:
        args['cat_feature'] = True
        args['save_path'] = f"{args['save_path']}/cat_feature"
    args['random_feature'] = False
    if args0.random_feature:
        args['random_feature'] = True
        args['save_path'] = f"{args['save_path']}/random_feature"
    args['structural_feature'] = False
    if args0.structural_feature:
        args['structural_feature'] = True
        args['save_path'] = f"{args['save_path']}/structural_feature"
    args['same_feature'] = False
    if args0.same_feature:
        args['same_feature'] = True
        args['save_path'] = f"{args['save_path']}/same_feature"
    args['drop_edges'] = False
    if args0.drop_edges:
        args['drop_edges'] = True
        args['save_path'] = f"{args['save_path']}/drop_edges"
    args['ego'] = False
    if args0.ego:
        args['ego'] = True
    args['add_edge_feature'] = False
    if args0.add_edge_feature:
        args['add_edge_feature'] = True
    args['num-layers'] = args0.num_layers
    args['thres'] = args0.thres
    args['comm_algo'] = args0.comm_algo
    
    print(args)
    final_results = []
    for r in range(cfg['runs']):
        args['seed'] = r
        # final_results.append(run_model(args))

        run_inference(args)
        break
        
    print('total time: ', time.time()-start_time)


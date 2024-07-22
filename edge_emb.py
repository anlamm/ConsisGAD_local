###### put this file together with main.py
###### python -u edge_emb.py --config config/yelp.yml --runs 1 --mode train


import argparse
import sys
import os
import csv
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from modules.data_loader import get_index_loader_test
from models import simpleGNN_MR
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

import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


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
        tmp_model = simpleGNN_MR(in_feats=args['node-in-dim'], hidden_feats=args['hidden-dim'], out_feats=args['node-out-dim'], 
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
        

def get_model_pred(model, graph, data_loader, sampler, args):
    model.eval()
    
    pred_list = []
    target_list = []
    with torch.no_grad():
        for node_idx in data_loader:
            _, _, blocks = sampler.sample_blocks(graph, node_idx.to(args['device']))
            
            pred = model(blocks)
            target = blocks[-1].dstdata['label']
            
            pred_list.append(pred.detach())
            target_list.append(target.detach())
        pred_list = torch.cat(pred_list, dim=0)
        target_list = torch.cat(target_list, dim=0)
        pred_list = pred_list.exp()[:, 1]
        
    return pred_list, target_list


def val_epoch(epoch, model, graph, valid_loader, test_loader, sampler, args):
    valid_dict = {}
    valid_pred, valid_target = get_model_pred(model, graph, valid_loader, sampler, args)
    v_roc, v_pr, _, _, _, _, v_f1, v_thre = eval_pred(valid_pred, valid_target)
    valid_dict['auc-roc'] = v_roc
    valid_dict['auc-pr'] = v_pr
    valid_dict['marco f1'] = v_f1
        
    test_dict = {}
    test_pred, test_target = get_model_pred(model, graph, test_loader, sampler, args)
    t_roc, t_pr, _, _, _, _, _, _ = eval_pred(test_pred, test_target)
    test_dict['auc-roc'] = t_roc
    test_dict['auc-pr'] = t_pr
    
    test_pred = test_pred.cpu().numpy()
    test_target = test_target.cpu().numpy()
    guessed_target = np.zeros_like(test_target)
    guessed_target[test_pred > v_thre] = 1
    t_f1 = f1_score(test_target, guessed_target, average='macro')
    test_dict['marco f1'] = t_f1
            
    return valid_dict, test_dict


def run_model(args):
    graph, label_loader, valid_loader, test_loader, unlabel_loader = get_index_loader_test(name=args['data-set'], 
                                                                                           batch_size=args['batch-size'], 
                                                                                           unlabel_ratio=args['unlabel-ratio'],
                                                                                           training_ratio=args['training-ratio'],
                                                                                           shuffle_train=args['shuffle-train'], 
                                                                                           to_homo=args['to-homo'])
    graph = graph.to(args['device'])
    
    args['node-in-dim'] = graph.ndata['feature'].shape[1]
    args['node-out-dim'] = 2
    
    my_model = create_model(args, graph.etypes)
    print(f"#Params: {num_params(my_model)}")
    
    if args['optim'] == 'adam':
        optimizer = optim.Adam(my_model.parameters(), lr=args['lr'], weight_decay=0.0)
    elif args['optim'] == 'rmsprop':
        optimizer = optim.RMSprop(my_model.parameters(), lr=args['lr'], weight_decay=0.0)
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num-layers'])
    
    train_epoch = UDA_train_epoch
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
            best_val = val_results['auc-roc']
            test_in_best_val = test_results
            
            if args['store-model']:
                m_utls.store_model(my_model, args)
                
    return list(test_in_best_val.values())


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def num_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params


def fetch_edge_emb(model, blocks: list, update_bn: bool=True):
    import dgl.function as fn
    model.eval()

    with torch.no_grad():

        final_num = blocks[-1].num_dst_nodes()
        h = blocks[0].srcdata['feature']
        h = model.dropout_in(h)
            
        inter_results = []
        h = model.proj_in(h)
            
        if model.in_bn is not None:
            h = model.in_bn(h, update_running_stats=update_bn)
            
        inter_results.append(h[:final_num])
        for block, gnn, bn in zip(blocks, model.gnn_list, model.bn_list):
            g = block
            features = h
            with g.local_scope():
                src_feats = dst_feats = features
                if g.is_block:
                    dst_feats = src_feats[:g.num_dst_nodes()]
                g.srcdata['h'] = src_feats
                g.dstdata['h'] = dst_feats
                
                for e_t in g.etypes:
                    g.apply_edges(gnn.udf_edges(e_t), etype=e_t)
                    
                # if gnn.bn_type in [2, 3]:
                #     if not gnn.multi_relation:
                #         g.edata['msg'] = gnn.edge_bn[gnn.e_types[0]](g.edata['msg'], update_running_stats=update_bn)
                #     else:
                #         for e_t in g.canonical_etypes:
                #             g.edata['msg'][e_t] = gnn.edge_bn[e_t[1]](g.edata['msg'][e_t], update_running_stats=update_bn)

                # etype_dict = {}
                # for e_t in g.etypes:
                #     etype_dict[e_t] = (fn.copy_e('msg', 'msg'), fn.sum('msg', 'out'))
                # g.multi_update_all(etype_dict=etype_dict, cross_reducer='stack')

                out = g.edata['msg']

        
        #### Prepare the plot_data
        plot_data = {}
        if not gnn.multi_relation:
            destlabels = g.dstdata['label']
            srclabels = g.srcdata['label']

            srcnodes, destnodes = g.edges()
            assert len(srcnodes) == len(out)

            edge_labels = torch.zeros(len(out))
            edge_labels[(srclabels[srcnodes] == 1) * (destlabels[destnodes] == 1)] = 1 ## AA
            edge_labels[(srclabels[srcnodes] == 0) * (destlabels[destnodes] == 1)] = 2 ## NA
            edge_labels[(srclabels[srcnodes] == 1) * (destlabels[destnodes] == 0)] = 3 ## AN
            edge_labels[(srclabels[srcnodes] == 0) * (destlabels[destnodes] == 0)] = 4 ## NN

            plot_data["1"] = {"emb": out.detach().cpu().tolist(), "label": edge_labels.detach().cpu().tolist()}
        else:
            destlabels = g.dstdata['label']
            srclabels = g.srcdata['label']

            for e_t in g.canonical_etypes:
                srcnodes, destnodes = g.edges(etype=e_t)

                assert len(srcnodes) == len(out[e_t])

                edge_labels = torch.zeros(len(out[e_t]))
                edge_labels[(srclabels[srcnodes] == 1) * (destlabels[destnodes] == 1)] = 1 ## AA
                edge_labels[(srclabels[srcnodes] == 0) * (destlabels[destnodes] == 1)] = 2 ## NA
                edge_labels[(srclabels[srcnodes] == 1) * (destlabels[destnodes] == 0)] = 3 ## AN
                edge_labels[(srclabels[srcnodes] == 0) * (destlabels[destnodes] == 0)] = 4 ## NN

                plot_data[e_t] = {"emb": out[e_t].detach().cpu().tolist(), "label": edge_labels.detach().cpu().tolist()}


        # ##### Prepare t-SNE coordinates
        # tsne_processor = TSNE(n_components=2, random_state=1234)
        # for etype, data in plot_data.items():
        #     print(f"============ TSNE {etype} =============")
        #     emb = data["emb"]
        #     tsne_z = tsne_processor.fit_transform(emb)
        #     plot_data[etype]["X_2d"] = tsne_z


        # ##### Dict to pandas
        # ### Columns: ["emb", "X_2d", "label", "etype"]
        # df = pd.DataFrame(columns=["emb", "X_2d", "label", "etype"])
        # for etype, data in plot_data.items():
        #     emb, tsne_z, label = data["emb"], data["X_2d"], data["label"]
        #     etype_list = [etype] * len(emb)
        #     df_single = pd.DataFrame({
        #         "emb": emb,
        #         "X_2d": tsne_z,
        #         "label": label, 
        #         "etype": etype_list,
        #     }, columns=["emb", "X_2d", "label", "etype"])

        #     df = pd.concat([df, df_single], axis=0)


        
        return plot_data
            



if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs. Default is 1.')
    parser.add_argument('--mode', type=str, default="train", help='which dataloader to use')
    args0 = parser.parse_args()
    cfg = vars(parser.parse_args())
    
    args = get_config(cfg['config'])
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:%d'%(args['device']))
    else:
        args['device'] = torch.device('cpu')
    args['mode'] = args0.mode                                     
    print(args)
    # final_results = []
    # for r in range(cfg['runs']):
    #     final_results.append(run_model(args))

    # args['batch-size'] = 1024


    #### Initialize graph
    graph, label_loader, valid_loader, test_loader, unlabel_loader = get_index_loader_test(name=args['data-set'], 
                                                                                           batch_size=args['batch-size'], 
                                                                                           unlabel_ratio=args['unlabel-ratio'],
                                                                                           training_ratio=args['training-ratio'],
                                                                                           shuffle_train=args['shuffle-train'], 
                                                                                           to_homo=args['to-homo'])
    graph = graph.to(args['device'])
    print(f"#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}")


    #### Initialize model
    args['node-in-dim'] = graph.ndata['feature'].shape[1]
    args['node-out-dim'] = 2
    
    model = create_model(args, graph.etypes)
    
    #### Load model checkpoint
    model.load_state_dict(torch.load(os.path.join('/home/yliumh/github/ConsisGAD-backup/ConsisGAD/model-weights',
                             args['data-set'] + '.pth')))
    

    #### Initialize sampler and dataloader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num-layers'])
    if args['mode'] == "train":
        data_loader = label_loader
    else:
        data_loader = test_loader

    #### Inference edge embeddings
    model.eval()
    e_emb_list = []
    target_list = []
    plot_data = {}
    with torch.no_grad():
        for node_idx in data_loader:
            _, _, blocks = sampler.sample_blocks(graph, node_idx.to(args['device']))
            
            plot_data_batch = fetch_edge_emb(model, blocks, update_bn=True)
            
            for etype, data in plot_data_batch.items():
                emb, label = data["emb"], data["label"]
                if etype not in plot_data:
                    plot_data[etype] = {
                        "emb": emb,
                        "label": label,
                    }
                else:
                    plot_data[etype]["emb"] = plot_data[etype]["emb"] + emb ## concat
                    plot_data[etype]["label"] = plot_data[etype]["label"] + label ## concat

    print(f"plot_data: {list(plot_data.keys())}")
    ##### To make four types of edges have same amount
    for e_t in plot_data.keys():
        emb = torch.FloatTensor(plot_data[e_t]["emb"])
        labels = torch.LongTensor(plot_data[e_t]["label"])
        text_labels = np.array(["", "AA", "NA", "AN", "NN"])

        AA_idx = torch.nonzero(torch.where(labels == 1, 1, 0), as_tuple=True)[0]
        NA_idx = torch.nonzero(torch.where(labels == 2, 1, 0), as_tuple=True)[0]
        AN_idx = torch.nonzero(torch.where(labels == 3, 1, 0), as_tuple=True)[0]
        NN_idx = torch.nonzero(torch.where(labels == 4, 1, 0), as_tuple=True)[0]


        min_num = min([AA_idx.shape[0], NA_idx.shape[0], AN_idx.shape[0], NN_idx.shape[0]])

            
        AA_idx = AA_idx[torch.randperm(AA_idx.shape[0])[:min_num]]
        NA_idx = NA_idx[torch.randperm(NA_idx.shape[0])[:min_num]]
        AN_idx = AN_idx[torch.randperm(AN_idx.shape[0])[:min_num]]
        NN_idx = NN_idx[torch.randperm(NN_idx.shape[0])[:min_num]]


        All_idx = torch.cat([AA_idx, NA_idx, AN_idx, NN_idx])
        All_idx = All_idx[torch.randperm(All_idx.shape[0])]

        # print(f"???: {emb.shape} {labels.shape} {torch.unique(labels)}")
        # print(f"!!!: {min_num}, {len(All_idx)}")
        # print(f"{AA_idx.shape} {NA_idx.shape} {AN_idx.shape} {NN_idx.shape}")
        # print(f"{AA_idx} {NA_idx} {AN_idx} {NN_idx}")

        # exit(0)

        plot_data[e_t]["emb"] = emb[All_idx].tolist()
        plot_data[e_t]["label"] = text_labels[labels[All_idx].numpy()].tolist()
    # tsne_processor = TSNE(n_components=2, random_state=1234)
    # for etype, data in plot_data.items():
    #     emb = data["emb"]
    #     print(f"============ TSNE {etype},{len(emb)},{args['mode']} =============")
    #     tsne_z = tsne_processor.fit_transform(np.array(emb))
    #     plot_data[etype]["x"] = tsne_z[:,0]
    #     plot_data[etype]["y"] = tsne_z[:,1]
    

    df = pd.DataFrame(columns=["emb", "label", "etype"])
    for etype, data in plot_data.items():
        emb, label = data["emb"], data["label"]
        etype_list = [etype] * len(emb)

        df_single = pd.DataFrame({
            "emb": emb,
            "label": label, 
            "etype": etype_list,
        }, columns=["emb", "label", "etype"])

        df = pd.concat([df, df_single], axis=0)
    print(df.head())
    df.to_csv(f"images/{args['data-set']}_{args['mode']}.csv", index=False) #### Save the data for futher use
    df = df.sample(frac=1).reset_index(drop=True) ### shuffle_rows

    #### plot
    print(f"============ Plotting =============")
    nsamples = 15000 #### Randomly sample n samples
    os.makedirs("images", exist_ok=True)
    if len(plot_data) > 1: ### multi-relation
        fig, axes = plt.subplots(1, len(plot_data), figsize=(5*len(plot_data), 5))

        for e_idx, etype in enumerate(list(plot_data.keys())):
            ax = axes[e_idx]

            df0 = df[df["etype"] == etype]
            if len(df0) > nsamples:
                df0 = df0.iloc[:nsamples, :]
            emb = df0["emb"].to_list()

            print(f"============ TSNE {etype},{len(emb)},{args['mode']} =============")
            tsne_processor = TSNE(n_components=2, random_state=1234)
            tsne_z = tsne_processor.fit_transform(np.array(emb))
            df0["x"] = tsne_z[:,0]
            df0["y"] = tsne_z[:,1]

            sns.scatterplot(df0, x="x", y="y", hue="label", palette="tab10", ax=ax)
            ax.set_title(f"{args['data-set']}, {etype}, {args['mode']}")
    else:
        fig = plt.figure(figsize=(5,5))
        ax = plt.gca()

        if len(df) > nsamples:
            df = df.iloc[:nsamples, :]
        emb = df["emb"].to_list()

        print(f"============ TSNE {etype},{len(emb)},{args['mode']} =============")
        tsne_processor = TSNE(n_components=2, random_state=1234)
        tsne_z = tsne_processor.fit_transform(np.array(emb))
        df["x"] = tsne_z[:,0]
        df["y"] = tsne_z[:,1]    
        sns.scatterplot(df, x="x", y="y", hue="label", palette="tab10", ax=ax)

        ax.set_title(f"{args['data-set']}, {args['mode']}")

    fig.savefig(f"images/{args['data-set']}_{args['mode']}.png", bbox_inches="tight")
    print('total time: ', time.time()-start_time)
    

### copied from baseline.py

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

from dgl.data.utils import load_graphs
from torch.utils.data import DataLoader as torch_dataloader


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

    if args["load_model"]:
        print(f"Loading pretrained checkpoint: {args['save_path']}/{args['data-set']}-{args['seed']}.pth")
        tmp_model.load_state_dict(torch.load(f"{args['save_path']}/{args['data-set']}-{args['seed']}.pth"))
    else:
        raise Exception("Must load checkpoint before eval")
    tmp_model.to(args['device'])
            
    return tmp_model


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


def plot_failures(epoch, model, graph, valid_loader, test_loader, sampler, args):
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

    #### test set
    preds = guessed_target
    labels = test_target


    failure_mask = (preds != labels)
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    nfail = failure_mask.sum()
    npos = pos_mask.sum()
    nneg = neg_mask.sum()
    npos_fail = (pos_mask * failure_mask).sum()
    nneg_fail = (neg_mask * failure_mask).sum()


    ### compute node degrees
    nodeids = []
    degs = []
    graph = graph.cpu()
    for bns in test_loader:
        nodeids += bns.cpu().tolist()
        bds = torch.zeros_like(bns)
        for etype in graph.etypes:
            bds += graph.in_degrees(bns, etype=etype)

        degs += bds.cpu().tolist()

    plot_data = {
        'degs': np.array(degs),
        'fail': failure_mask,
        'label': labels, 
        'nodeids': np.array(nodeids),
    }
    plot_data = pd.DataFrame(plot_data, columns=list(plot_data.keys()))
    plot_data = plot_data[plot_data['fail'] == True]

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    ax = axes[0]
    sns.histplot(data=plot_data[plot_data["label"]==1], x="degs", stat="proportion", color="orange", alpha = 0.8, bins=50, ax=ax)
    ax.set_title(f"{args['data-set']}, abnormal ({npos_fail}/{npos})")

    ax = axes[1]
    sns.histplot(data=plot_data[plot_data["label"]==0], x="degs", stat="proportion", alpha = 0.8, bins=50, ax=ax)
    ax.set_title(f"{args['data-set']}, normal ({nneg_fail}/{nneg})")


    savedir = 'images'
    if args['random_feature']:
        savedir = f'{savedir}/random_feature'
    if args['structural_feature']:
        savedir = f'{savedir}/random_feature'
    if args['drop_edges']:
        savedir = f'{savedir}/drop_edges'
    os.makedirs(f'{savedir}', exist_ok=True)
    fig.savefig(f'{savedir}/failure_{args["data-set"]}.png', bbox_inches='tight')

    return valid_dict, test_dict


def mark_failures_mask(epoch, model, graph, valid_loader, test_loader, sampler, args):

    n = graph.number_of_nodes()
    preds = np.zeros(n)
    labels = graph.ndata['label'].cpu().numpy()

    valid_pred, valid_target = get_model_pred(model, graph, valid_loader, sampler, args)
    v_roc, v_pr, _, _, _, _, v_f1, v_thre = eval_pred(valid_pred, valid_target) ### 确定置信度划分阈值
    valid_pred = valid_pred.cpu().numpy()
    valid_target = valid_target.cpu().numpy()
    guessed_target = np.zeros_like(valid_target)
    guessed_target[valid_pred > v_thre] = 1
    valid_mask = graph.ndata['val_mask'].nonzero(as_tuple=True)[0].cpu().numpy()
    preds[valid_mask] = guessed_target
    labels[valid_mask] = valid_target
    ###
    failure_mask = (guessed_target != valid_target)
    pos_mask = (valid_target == 1)
    neg_mask = (valid_target == 0)
    npos = pos_mask.sum()
    nneg = neg_mask.sum()
    npos_fail = (pos_mask * failure_mask).sum()
    nneg_fail = (neg_mask * failure_mask).sum()
    print(f"#Valid failures: pos {npos_fail}/{npos}, neg {nneg_fail}/{nneg}")
        

    test_pred, test_target = get_model_pred(model, graph, test_loader, sampler, args)
    test_pred = test_pred.cpu().numpy()
    test_target = test_target.cpu().numpy()
    guessed_target = np.zeros_like(test_target)
    guessed_target[test_pred > v_thre] = 1
    test_mask = graph.ndata['test_mask'].nonzero(as_tuple=True)[0].cpu().numpy()
    preds[test_mask] = guessed_target
    labels[test_mask] = test_target
    ###
    failure_mask = (guessed_target != test_target)
    pos_mask = (test_target == 1)
    neg_mask = (test_target == 0)
    npos = pos_mask.sum()
    nneg = neg_mask.sum()
    npos_fail = (pos_mask * failure_mask).sum()
    nneg_fail = (neg_mask * failure_mask).sum()
    print(f"#Test failures: pos {npos_fail}/{npos}, neg {nneg_fail}/{nneg}")

    train_nids = graph.ndata['train_mask'].nonzero(as_tuple=True)[0]
    train_loader = torch_dataloader(train_nids, batch_size=2**10, shuffle=False, drop_last=False, num_workers=0)
    train_pred, train_target = get_model_pred(model, graph, train_loader, sampler, args)
    train_pred = train_pred.cpu().numpy()
    train_target = train_target.cpu().numpy()
    guessed_target = np.zeros_like(train_target)
    guessed_target[train_pred > v_thre] = 1
    train_mask = graph.ndata['train_mask'].nonzero(as_tuple=True)[0].cpu().numpy()
    preds[train_mask] = guessed_target
    labels[train_mask] = train_target
    ###
    failure_mask = (guessed_target != train_target)
    pos_mask = (train_target == 1)
    neg_mask = (train_target == 0)
    npos = pos_mask.sum()
    nneg = neg_mask.sum()
    npos_fail = (pos_mask * failure_mask).sum()
    nneg_fail = (neg_mask * failure_mask).sum()
    print(f"#Train failures: pos {npos_fail}/{npos}, neg {nneg_fail}/{nneg}")


    failure_mask = (preds != labels)
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    npos = pos_mask.sum()
    nneg = neg_mask.sum()
    npos_fail = (pos_mask * failure_mask).sum()
    nneg_fail = (neg_mask * failure_mask).sum()
    print(f"#ALL failures: pos {npos_fail}/{npos}, neg {nneg_fail}/{nneg}")


    ### write into dglgraph.ndata
    raw_dir = "data/offline"
    name = f"{args['data-set']}"
    outdir = f"{raw_dir}/addfailure"
    os.makedirs(outdir, exist_ok=True)

    data, _ = load_graphs(os.path.join(raw_dir, f'{name}.dglgraph'))
    graph = data[0]

    graph.ndata['failure_ConsisGNN'] = torch.LongTensor(failure_mask)
    
    labels2 = graph.ndata['label'].cpu().numpy()
    assert (labels2 - labels).sum() < 1e-10

    dgl.save_graphs(f"{outdir}/{name}_added_failure.dglgraph", [graph])

    print(f"New DGL graph saved to {outdir}/{name}_added_failure.dglgraph")
    print(f"New ndata.keys: {list(graph.ndata.keys())}")




def run_model(args):
    if args['data-set'] in ['amazon', 'yelp', 'tfinance']:
        graph, label_loader, valid_loader, test_loader, unlabel_loader = get_index_loader_test(name=args['data-set'], 
                                                                                            batch_size=args['batch-size'], 
                                                                                            unlabel_ratio=args['unlabel-ratio'],
                                                                                            training_ratio=args['training-ratio'],
                                                                                            shuffle_train=args['shuffle-train'], 
                                                                                            to_homo=args['to-homo'],
                                                                                            random_feature=args['random_feature'],
                                                                                            structural_feature=args['structural_feature'],
                                                                                            verbose=args['debug'],
                                                                                            load_offline=True,
                                                                                            seed = args['seed'])

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
        
        my_model = create_model(args, graph.etypes)
        print(f"#Params: {num_params(my_model)}")
        
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num-layers'])
        
        # val_results, test_results = val_epoch(0, my_model, graph, valid_loader, test_loader, sampler, args)
        # plot_failures(0, my_model, graph, valid_loader, test_loader, sampler, args)
        mark_failures_mask(0, my_model, graph, valid_loader, test_loader, sampler, args)

    elif args['data-set'] == 'merge':
        graph, label_loader, valid_loader, test_loaders, unlabel_loader, labeled_loaders, _ = get_index_loader_test(name=args['data-set'],   ##### dataset="merge"
                                                                                           batch_size=args['batch-size'], 
                                                                                           unlabel_ratio=args['unlabel-ratio'],
                                                                                           training_ratio=args['training-ratio'],
                                                                                           shuffle_train=args['shuffle-train'], 
                                                                                           to_homo=args['to-homo'],
                                                                                           fill=args['fill-mode'],
                                                                                           random_feature=args['random_feature'],
                                                                                           structural_feature=args['structural_feature'],
                                                                                           verbose=args['debug'],
                                                                                           load_offline=True,
                                                                                           seed = args['seed'])
        
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
        
        my_model = create_model(args, graph.etypes)
        print(f"#Params: {num_params(my_model)}")
        
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num-layers'])
        
        # val_results, test_results = val_epoch(0, my_model, graph, valid_loader, test_loader, sampler, args)
        datanames = ['yelp', 'amazon', 'tfinance']
        for i, name in enumerate(datanames):
            args['data-set'] = f"{name}-merge"
            plot_failures(0, my_model, graph, valid_loader, test_loaders[i], sampler, args)

    else:
        raise NotImplementedError
    



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
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--save_path', type=str, default="model-weights", help="path for saving model weights")
    parser.add_argument('--drop_edges', action="store_true")
    parser.add_argument('--load_model', action="store_true")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--fill_mode', type=str, default="zero", choices=["zero", "mean"])
    args0 = parser.parse_args()
    cfg = vars(parser.parse_args())
    
    args = get_config(cfg['config'])
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:%d'%(args['device']))
    else:
        args['device'] = torch.device('cpu')

    # args['epochs'] = 300
    args['debug'] = False
    if args0.debug:
        args['debug'] = True
    args['save_path'] = args0.save_path
    args['random_feature'] = False
    if args0.random_feature:
        args['random_feature'] = True
        args['save_path'] = f"{args['save_path']}/random_feature"
    args['structural_feature'] = False
    if args0.structural_feature:
        args['structural_feature'] = True
        args['save_path'] = f"{args['save_path']}/structural_feature"
    args['drop_edges'] = False
    if args0.drop_edges:
        args['drop_edges'] = True
        args['save_path'] = f"{args['save_path']}/drop_edges"

    args['seed'] = 0
    if args0.load_model:
        args['load_model'] = True
    else:
        args['load_model'] = False
    if args0.dataset is not None:
        args['data-set'] = args0.dataset
    args['fill-mode'] = args0.fill_mode
    print(args)

    run_model(args)



    
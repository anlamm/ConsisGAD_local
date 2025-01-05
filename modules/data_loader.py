import torch
import dgl
import os
import numpy as np
from dgl.data.utils import load_graphs
from torch.utils.data import DataLoader as torch_dataloader
from dgl.dataloading import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import dgl.function as fn
import logging
import pickle
import os


##### 
import pandas as pd
import networkx as nx

    
def get_dataset(name: str, raw_dir: str, to_homo: bool=False, random_state: int=717):
    if name == 'yelp':
        yelp_data = dgl.data.FraudYelpDataset(raw_dir=raw_dir, random_seed=7537, verbose=False)
        graph = yelp_data[0]
        if to_homo:
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)

    elif name == 'amazon':
        amazon_data = dgl.data.FraudAmazonDataset(raw_dir=raw_dir, random_seed=7537, verbose=False)
        graph = amazon_data[0]
        if to_homo:
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
            
    elif name == 'tsocial':
        t_social, _ = load_graphs(os.path.join(raw_dir, 'tsocial'))
        graph = t_social[0]
        graph.ndata['feature'] = graph.ndata['feature'].float()
        
    elif name == 'tfinance':
        t_finance, _ = load_graphs(os.path.join(raw_dir, 'tfinance'))
        graph = t_finance[0]
        # graph.ndata['label'] = graph.ndata['label'].argmax(1)
        graph.ndata['feature'] = graph.ndata['feature'].float()

    elif name in ['weibo', 'reddit', 'tolokers', 'yelp', 'questions', 'elliptic',  'dgraphfin']:
        data, _ = load_graphs(os.path.join(raw_dir, name))
        graph = data[0]
        # graph.ndata['label'] = graph.ndata['label'].argmax(1)
        graph.ndata['feature'] = graph.ndata['feature'].float()

    else:
        raise
    
    return graph

    
def get_index_loader_test(name: str, batch_size: int, unlabel_ratio: int=1, training_ratio: float=-1,
                             shuffle_train: bool=True, to_homo:bool=False, fill="zero", random_feature=False, structural_feature=False, same_feature=False, cat_feature=False, verbose=False, load_offline=False, seed=None, add_edge_feature=False):
    # assert name in ['yelp', 'amazon', 'tfinance', 'tsocial', 'merge'], 'Invalid dataset name'

    if load_offline:
        graph = load_graphs(os.path.join("data", "offline", f"{name}.dglgraph"))[0][0]
        if structural_feature:
            graph = load_graphs(os.path.join("data", "offline", f"{name}_added_embs.dglgraph"))[0][0]

            if cat_feature:
                graph.ndata['feature'] = torch.cat([graph.ndata['feature'], graph.ndata['deepwalk']], axis=1)  ### use deepwalk structural embeddings
            else:
                graph.ndata['feature'] = graph.ndata['deepwalk']  ### use deepwalk structural embeddings

        graph.ndata['train_mask'] = graph.ndata['train_masks'][:,seed]  ### Use pre-generated splits
        graph.ndata['val_mask'] = graph.ndata['val_masks'][:,seed]
        graph.ndata['test_mask'] = graph.ndata['test_masks'][:,seed]
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']


        train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
        valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=True)[0]
        test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=True)[0]


        labeled_nids = train_nids
        unlabeled_nids = np.concatenate([valid_nids, test_nids, train_nids])
        
        power = 10 if name == 'tfinance' or name == 'merge' else 16
        power = 4
        
        valid_loader = torch_dataloader(valid_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
        test_loader = torch_dataloader(test_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
        labeled_loader = torch_dataloader(labeled_nids, batch_size=batch_size, shuffle=shuffle_train, drop_last=True, num_workers=0)
        unlabeled_loader = torch_dataloader(unlabeled_nids, batch_size=batch_size * unlabel_ratio, shuffle=shuffle_train, drop_last=True, num_workers=0)


        if random_feature:
            # n, d = graph.ndata['feature'].shape
            # new_features = torch.randn(n, 32).float()
            # graph.ndata['feature'] = new_features
            if cat_feature:
                graph.ndata['feature'] = torch.cat([graph.ndata['feature'], graph.ndata['random_feature']], axis=1)  ### use deepwalk structural embeddings
            else:
                graph.ndata['feature'] = graph.ndata['random_feature']
        if same_feature:
            if cat_feature:
                graph.ndata['feature'] = torch.cat([graph.ndata['feature'], graph.ndata['same_feature']], axis=1)  ### use same feature for each node
            else:
                graph.ndata['feature'] = graph.ndata['same_feature']
            

        if verbose:
            nnodes = graph.number_of_nodes()
            print(f"{name}: #Train: {train_mask.sum()}({train_mask.sum()*100.0/nnodes:.1f}%), #Val: {val_mask.sum()}({val_mask.sum()*100.0/nnodes:.1f}%), #Test: {test_mask.sum()}({test_mask.sum()*100.0/nnodes:.1f}%)")

        if name == "merge":
            graph_ids = torch.unique(graph.ndata['graph_id'])
            train_mask = graph.ndata['train_mask']
            val_mask = graph.ndata['val_mask']
            test_mask = graph.ndata['test_mask']
            
            labeled_loaders, valid_loaders, test_loaders  = [], [], []
            for graph_id in graph_ids:
                sub_train_nids = torch.nonzero((graph.ndata['graph_id'] == graph_id) * train_mask, as_tuple=True)[0]
                sub_valid_nids = torch.nonzero((graph.ndata['graph_id'] == graph_id) * val_mask, as_tuple=True)[0]
                sub_test_nids = torch.nonzero((graph.ndata['graph_id'] == graph_id) * test_mask, as_tuple=True)[0]

                labeled_loaders.append(torch_dataloader(sub_train_nids, batch_size=batch_size, shuffle=shuffle_train, drop_last=True, num_workers=0))
                valid_loaders.append(torch_dataloader(sub_valid_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4))
                test_loaders.append(torch_dataloader(sub_test_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4))

            return graph, labeled_loader, valid_loader, test_loaders, unlabeled_loader, labeled_loaders, valid_loaders
        

        if add_edge_feature:
            for et in graph.etypes:
                graph.edges[et].data['eh'] = torch.ones(len(graph.edges(etype=et)[0]), 4)

        graph = graph.long()
        return graph, labeled_loader, valid_loader, test_loader, unlabeled_loader




    if name == "merge":
        names = ['yelp', 'amazon', 'tfinance']
        graphs = []

        t_n = 0
        for name in names:
            graph = get_dataset(name, 'data/', to_homo=True, random_state=7537)
            t_n += graph.num_nodes()
        train_mask = torch.zeros(t_n).bool()
        val_mask = torch.zeros(t_n).bool()
        test_mask = torch.zeros(t_n).bool()

        c_n = 0
        valid_nids_m, test_nids_m, labeled_nids_m, unlabeled_nids_m = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
        labeled_nids_s = []
        valid_nids_s = []
        labels_m = np.array([])
        features_list = []
        test_loaders = []
        for n_idx, name in enumerate(names):
            graph = get_dataset(name, 'data/', to_homo=True, random_state=7537)
            index = np.arange(graph.num_nodes())
            labels = graph.ndata['label']
            if name == 'amazon':
                index = np.arange(3305, graph.num_nodes())
            index += c_n  ###### already have c_n nodes before this graph
            c_n += graph.num_nodes()
            graph.ndata['feature'] = graph.ndata['feature'].float()
            graph.ndata['graph_id'] = torch.LongTensor([n_idx]).repeat(graph.number_of_nodes())
            graph.ndata['number_of_nodes'] = torch.LongTensor([graph.number_of_nodes()]).repeat(graph.number_of_nodes())
            
            train_nids, valid_test_nids = train_test_split(index, stratify=labels[index-c_n],
                                                        train_size=training_ratio/100., random_state=2, shuffle=True)
            valid_nids, test_nids = train_test_split(valid_test_nids, stratify=labels[valid_test_nids-c_n],
                                                    test_size=0.67, random_state=2, shuffle=True)   

            train_mask[train_nids] = 1
            val_mask[valid_nids] = 1
            test_mask[test_nids] = 1
            
            labeled_nids_m = np.concatenate([labeled_nids_m, train_nids])
            labeled_nids_s.append(train_nids)
            unlabeled_nids_m = np.concatenate([unlabeled_nids_m, valid_nids, test_nids, train_nids])

            valid_nids_m = np.concatenate([valid_nids_m, valid_nids])
            valid_nids_s.append(valid_nids)
            test_nids_m = np.concatenate([test_nids_m, test_nids])

            labels_m = np.concatenate([labels_m, labels.numpy()])
            features_list.append(graph.ndata['feature'])

            graphs.append(graph)


            if verbose:
                print(f"============= {name} =============")
                print(graph)
                print(list(graph.ndata.keys()))
                print(list(graph.edata.keys()))
                print(f"==================================")

            power = 10
            test_loader = torch_dataloader(test_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
            test_loaders.append(test_loader)     ####### split different datasets, test each dataset individually

        def diag_merge_features(features_list, fill_mode="zero"):
            #### [feat_list1, feat_list2, feat_list3]
            total_feat_dim = 0
            for feat_tensor in features_list:
                total_feat_dim += feat_tensor.shape[1]

            c_d = 0
            ret_features_list = []
            for feat_tensor in features_list:
                n, d = feat_tensor.shape
                ret_feat = torch.zeros((n, total_feat_dim)).float()
                ret_feat[:, c_d:c_d+d] = feat_tensor
                c_d += d

                ret_features_list.append(ret_feat)

            assert fill_mode in ["zero", "mean"]
            if fill_mode == "mean":
                means = []
                for ret_feat in ret_features_list:
                    means.append(ret_feat.mean(0))
                for i in range(len(ret_features_list)):
                    for j in range(len(ret_features_list)):
                        if j == i:
                            continue
                        ret_features_list[i] += means[j].repeat(ret_features_list[i].shape[0], 1)

            return ret_features_list

        features = diag_merge_features(features_list, fill_mode=fill)
        for i in range(len(graphs)):
            graphs[i].ndata['feature'] = features[i]

        ####### Before dgl.batch to merge graphs, make sure only exists two ndata: feature and label, and NO edata
        for i in range(len(graphs)):
            graph = graphs[i]
            ndata_keys = list(graph.ndata.keys())
            for key in ndata_keys:
                if key not in ['feature', 'label', 'graph_id', 'number_of_nodes']:
                    graph.ndata.pop(key)
            
            edata_keys = list(graph.edata.keys())
            for key in edata_keys:
                graph.edata.pop(key)
            
            graphs[i] = graph
        
        ####### Merge different graphs
        graph = dgl.batch(graphs)
        

        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        graph.ndata['label'] = torch.LongTensor(labels_m)

        if verbose:
            print(graph)
            print(list(graph.ndata.keys()))
            print(list(graph.edata.keys()))

            #### check train/val/test splits
            subgraph_idx = graph.ndata['graph_id']
            subgraph_nnodes = graph.ndata['number_of_nodes']
            train_mask = graph.ndata['train_mask']
            val_mask = graph.ndata['val_mask']
            test_mask = graph.ndata['test_mask']

            for n_idx, name in enumerate(names):
                node_idx = torch.nonzero(torch.where(subgraph_idx == n_idx, 1, 0), as_tuple=True)[0]
                nnodes = subgraph_nnodes[node_idx]
                assert torch.unique(nnodes).shape[0] == 1
                nnodes = nnodes[0]

                print(f"{name}: #Train: {train_mask[node_idx].sum()}({train_mask[node_idx].sum()*100.0/nnodes:.1f}%), #Val: {val_mask[node_idx].sum()}({val_mask[node_idx].sum()*100.0/nnodes:.1f}%), #Test: {test_mask[node_idx].sum()}({test_mask[node_idx].sum()*100.0/nnodes:.1f}%)")


        power = 10
        
        valid_loader = torch_dataloader(valid_nids_m, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
        test_loader = torch_dataloader(test_nids_m, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
        labeled_loader = torch_dataloader(labeled_nids_m, batch_size=batch_size, shuffle=shuffle_train, drop_last=True, num_workers=0)
        unlabeled_loader = torch_dataloader(unlabeled_nids_m, batch_size=batch_size * unlabel_ratio, shuffle=shuffle_train, drop_last=True, num_workers=0)

        labeled_loaders = [] ### train loaders
        for labeled_nids in labeled_nids_s:
            labeled_loaders.append(torch_dataloader(labeled_nids, batch_size=batch_size, shuffle=shuffle_train, drop_last=True, num_workers=0))
        valid_loaders = [] 
        for valid_nids in valid_nids_s:
            valid_loaders.append(torch_dataloader(valid_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4))



        if random_feature:
            n, d = graph.ndata['feature'].shape
            new_features = torch.randn(n, 32).float()
            graph.ndata['feature'] = new_features

        return graph, labeled_loader, valid_loader, test_loaders, unlabeled_loader, labeled_loaders, valid_loaders
        
    else:
    
        graph = get_dataset(name, 'data/', to_homo=to_homo, random_state=7537)
        
        index = np.arange(graph.num_nodes())
        labels = graph.ndata['label']
        if name == 'amazon':
            index = np.arange(3305, graph.num_nodes())
        
        train_nids, valid_test_nids = train_test_split(index, stratify=labels[index],
                                                    train_size=training_ratio/100., random_state=2, shuffle=True)
        valid_nids, test_nids = train_test_split(valid_test_nids, stratify=labels[valid_test_nids],
                                                test_size=0.67, random_state=2, shuffle=True)
        
        train_mask = torch.zeros_like(labels).bool()
        val_mask = torch.zeros_like(labels).bool()
        test_mask = torch.zeros_like(labels).bool()

        train_mask[train_nids] = 1
        val_mask[valid_nids] = 1
        test_mask[test_nids] = 1
        
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        
        labeled_nids = train_nids
        unlabeled_nids = np.concatenate([valid_nids, test_nids, train_nids])
        
        power = 10 if name == 'tfinance' else 16
        
        valid_loader = torch_dataloader(valid_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
        test_loader = torch_dataloader(test_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
        labeled_loader = torch_dataloader(labeled_nids, batch_size=batch_size, shuffle=shuffle_train, drop_last=True, num_workers=0)
        unlabeled_loader = torch_dataloader(unlabeled_nids, batch_size=batch_size * unlabel_ratio, shuffle=shuffle_train, drop_last=True, num_workers=0)


        if random_feature:
            n, d = graph.ndata['feature'].shape
            new_features = torch.randn(n, 32).float()
            graph.ndata['feature'] = new_features

        if verbose:
            nnodes = graph.number_of_nodes()
            print(f"{name}: #Train: {train_mask.sum()}({train_mask.sum()*100.0/nnodes:.1f}%), #Val: {val_mask.sum()}({val_mask.sum()*100.0/nnodes:.1f}%), #Test: {test_mask.sum()}({test_mask.sum()*100.0/nnodes:.1f}%)")

        return graph, labeled_loader, valid_loader, test_loader, unlabeled_loader
    



def offline_data_split(training_ratio, to_homo=False, nseeds=5, fill_mode="zero"): ### seeds: number of seeds

    root = "data"
    root = "/home/yliumh/github/GADBench/datasets"
    outdir = f"{root}/offline"
    outdir = f"data/offline"
    os.makedirs(outdir, exist_ok=True)

    #### single dataset
    print(f"================ Single dataset =================")
    datanames = ['yelp', 'amazon', 'tfinance']
    # datanames = ['weibo', 'reddit', 'tolokers', 'yelp', 'questions', 'elliptic',  'dgraphfin', 'tsocial']

    for name in datanames:
        graph = get_dataset(name, f'{root}/', to_homo=to_homo, random_state=7537) ## random_state not used
        n = graph.number_of_nodes()
        graph.ndata['feature'] = graph.ndata['feature'].float()

        train_masks = torch.zeros([n,nseeds]).bool()
        val_masks = torch.zeros([n,nseeds]).bool()
        test_masks = torch.zeros([n,nseeds]).bool()
        for seed in np.arange(nseeds):
        
            index = np.arange(graph.num_nodes())
            labels = graph.ndata['label']
            if name == 'amazon':
                index = np.arange(3305, graph.num_nodes())
            
            train_nids, valid_test_nids = train_test_split(index, stratify=labels[index],
                                                        train_size=training_ratio/100., random_state=seed, shuffle=True)
            valid_nids, test_nids = train_test_split(valid_test_nids, stratify=labels[valid_test_nids],
                                                    test_size=0.67, random_state=seed, shuffle=True)

            train_masks[train_nids, seed] = 1
            val_masks[valid_nids, seed] = 1
            test_masks[test_nids, seed] = 1

        graph.ndata['train_masks'] = train_masks
        graph.ndata['val_masks'] = val_masks
        graph.ndata['test_masks'] = test_masks

        n, d = graph.ndata['feature'].shape
        new_features = torch.randn(n, 32).float()
        graph.ndata['random_feature'] = new_features

        new_features = torch.ones(n, 32).float()
        graph.ndata['same_feature'] = new_features

        print(f"{name}: {graph.ndata['train_masks'].sum(0)}\n {graph.ndata['val_masks'].sum(0)}\n {graph.ndata['test_masks'].sum(0)}")
        print(f"{name}: {graph.ndata['train_masks'].sum(0)*100.0/n}%\n {graph.ndata['val_masks'].sum(0)*100.0/n}%\n {graph.ndata['test_masks'].sum(0)*100.0/n}%")
    
        dgl.save_graphs(f"{outdir}/{name}.dglgraph", [graph])


    #### merged dataset
    print(f"================ Merged dataset =================")
    graphs = []

    t_n = 0
    for name in datanames:
        graph = get_dataset(name, f'{root}/', to_homo=True, random_state=7537)
        t_n += graph.num_nodes()

    train_masks = torch.zeros(t_n, nseeds).bool()
    val_masks = torch.zeros(t_n, nseeds).bool()
    test_masks = torch.zeros(t_n, nseeds).bool()

    c_n = 0
    features_list = []
    for n_idx, name in enumerate(datanames):
        graph = get_dataset(name, f'{root}/', to_homo=True, random_state=7537)
        index = np.arange(graph.num_nodes())
        labels = graph.ndata['label']
        if name == 'amazon':
            index = np.arange(3305, graph.num_nodes())
        index += c_n  ###### already have c_n nodes before this graph
        c_n += graph.num_nodes()
        graph.ndata['feature'] = graph.ndata['feature'].float()
        graph.ndata['graph_id'] = torch.LongTensor([n_idx]).repeat(graph.number_of_nodes())
        graph.ndata['number_of_nodes'] = torch.LongTensor([graph.number_of_nodes()]).repeat(graph.number_of_nodes())
        features_list.append(graph.ndata['feature'])
        graphs.append(graph)
        
        for seed in np.arange(nseeds):
            train_nids, valid_test_nids = train_test_split(index, stratify=labels[index-c_n],
                                                            train_size=training_ratio/100., random_state=seed, shuffle=True)
            valid_nids, test_nids = train_test_split(valid_test_nids, stratify=labels[valid_test_nids-c_n],
                                                        test_size=0.67, random_state=seed, shuffle=True)   

            train_masks[train_nids, seed] = 1
            val_masks[valid_nids, seed] = 1
            test_masks[test_nids, seed] = 1
            
    def diag_merge_features(features_list, fill_mode="zero"):
        #### [feat_list1, feat_list2, feat_list3]
        total_feat_dim = 0
        for feat_tensor in features_list:
            total_feat_dim += feat_tensor.shape[1]

        c_d = 0
        ret_features_list = []
        for feat_tensor in features_list:
            n, d = feat_tensor.shape
            ret_feat = torch.zeros((n, total_feat_dim)).float()
            ret_feat[:, c_d:c_d+d] = feat_tensor
            c_d += d

            ret_features_list.append(ret_feat)

        assert fill_mode in ["zero", "mean"]
        if fill_mode == "mean":
            means = []
            for ret_feat in ret_features_list:
                means.append(ret_feat.mean(0))
            for i in range(len(ret_features_list)):
                for j in range(len(ret_features_list)):
                    if j == i:
                        continue
                    ret_features_list[i] += means[j].repeat(ret_features_list[i].shape[0], 1)

        return ret_features_list


    features = diag_merge_features(features_list, fill_mode=fill_mode)
    for i in range(len(graphs)):
        graphs[i].ndata['feature'] = features[i]

    ####### Before dgl.batch to merge graphs, make sure only exists two ndata: feature and label, and NO edata
    for i in range(len(graphs)):
        graph = graphs[i]
        ndata_keys = list(graph.ndata.keys())
        for key in ndata_keys:
            if key not in ['feature', 'label', 'graph_id', 'number_of_nodes']:
                graph.ndata.pop(key)
            
        edata_keys = list(graph.edata.keys())
        for key in edata_keys:
            graph.edata.pop(key)
            
        graphs[i] = graph
        
    ####### Merge different graphs
    graph = dgl.batch(graphs)
        

    graph.ndata['train_masks'] = train_masks
    graph.ndata['val_masks'] = val_masks
    graph.ndata['test_masks'] = test_masks


    n, d = graph.ndata['feature'].shape
    new_features = torch.randn(n, 32).float()
    graph.ndata['random_feature'] = new_features

    new_features = torch.ones(n, 32).float()
    graph.ndata['same_feature'] = new_features

    print(f"merge: {graph.ndata['train_masks'].sum(0)}\n {graph.ndata['val_masks'].sum(0)}\n {graph.ndata['test_masks'].sum(0)}")
    print(f"merge: {graph.ndata['train_masks'].sum(0)*100.0/n}%\n {graph.ndata['val_masks'].sum(0)*100.0/n}%\n {graph.ndata['test_masks'].sum(0)*100.0/n}%")
    
    dgl.save_graphs(f"{outdir}/merge.dglgraph", [graph])
    
    #### check train/val/test splits
    subgraph_idx = graph.ndata['graph_id']
    subgraph_nnodes = graph.ndata['number_of_nodes']
    train_masks = graph.ndata['train_masks']
    val_masks = graph.ndata['val_masks']
    test_masks = graph.ndata['test_masks']
    for n_idx, name in enumerate(datanames):
        node_idx = torch.nonzero(torch.where(subgraph_idx == n_idx, 1, 0), as_tuple=True)[0]
        nnodes = subgraph_nnodes[node_idx]
        assert torch.unique(nnodes).shape[0] == 1
        nnodes = nnodes[0]

        print(f"{name}: #Train: {train_masks[node_idx].sum(0)}, #Val: {val_masks[node_idx].sum(0)}, #Test: {test_masks[node_idx].sum(0)}")
        print(f"{name}: #Train: {train_masks[node_idx].sum(0)*100.0/nnodes}%, #Val: {val_masks[node_idx].sum(0)*100.0/nnodes}%, #Test: {test_masks[node_idx].sum(0)*100.0/nnodes}%")


class OnlineLCLoader(torch_dataloader):
    def __init__(self, root_nodes, graph, feats, labels=None, drop_edge_rate=0, **kwargs):
        self.graph = graph
        self.labels = labels
        self._drop_edge_rate = drop_edge_rate
        self.ego_graph_nodes = root_nodes
        self.feats = feats
        self.device = self.graph.device

        dataset = np.arange(len(root_nodes))
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def __collate_fn__(self, batch_idx):
        ego_nodes = [self.ego_graph_nodes[i] for i in batch_idx]
        subgs = [self.graph.subgraph(torch.LongTensor(ego_nodes[i]).to(self.device), relabel_nodes=True) for i in range(len(ego_nodes))]
        sg = dgl.batch(subgs)

        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long().to(self.device)
        num_nodes = [x.shape[0] for x in ego_nodes]
        num_nodes = torch.LongTensor([0] + num_nodes).to(self.device)
        # cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]

        # sg = sg.remove_self_loop().add_self_loop()
        sg.ndata["feat"] = self.feats[nodes]
        # targets = torch.from_numpy(cum_num_nodes)
        targets = torch.cumsum(num_nodes, dim=0)[:-1]
        
        # return sg, targets, label, nodes
        return sg, targets

def load_ego_graphs(name, seed, size=256):
    path = f"data/offline/subgraphs/{name}.dglgraph-lc-ego-graphs-{size}-{seed}.pt"
    nodes = torch.load(path)
    
    return nodes

def setup_training_dataloder(loader_type, training_nodes, graph, feats, batch_size, drop_edge_rate=0, pretrain_clustergcn=False, cluster_iter_data=None):
    num_workers = 0

    if loader_type == "lc":
        assert training_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")
    
    # print(" -------- drop edge rate: {} --------".format(drop_edge_rate))
    dataloader = OnlineLCLoader(training_nodes, graph, feats=feats, drop_edge_rate=drop_edge_rate, batch_size=batch_size, shuffle=True, drop_last=True, persistent_workers=False, num_workers=num_workers)
    return dataloader


def setup_eval_dataloder(loader_type, graph, feats, ego_graph_nodes=None, batch_size=128, shuffle=False):
    num_workers = 0
    if loader_type == "lc":
        assert ego_graph_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    dataloader = OnlineLCLoader(ego_graph_nodes, graph, feats, batch_size=2**8, shuffle=shuffle, drop_last=False, persistent_workers=False, num_workers=num_workers)
    return dataloader


def get_dataset_offline(name: str, raw_dir: str, to_homo: bool=False, random_state: int=717):
    print(f'Loading graph from {os.path.join(raw_dir, name)}')
    data, _ = load_graphs(os.path.join(raw_dir, name))
    graph = data[0]
    if to_homo:
        graph = dgl.to_homogeneous(graph, ndata=list(graph.ndata.keys()))
        graph = dgl.add_self_loop(graph)
    # graph.ndata['label'] = graph.ndata['label'].argmax(1)
    graph.ndata['feature'] = graph.ndata['feature'].float()
    return graph

def dglgraph2CSV(nnodes=None, nhop=None):
    
    root = "data/offline/addfailure"
    # root = "/home/yliumh/github/GADBench/datasets"
    outdir = f"{root}/CSV"
    os.makedirs(outdir, exist_ok=True)


    
    datanames = ['yelp', 'amazon', 'tfinance']

    for name in datanames:
        print(f"{name}")

        # graph = get_dataset(name, f'{root}/', to_homo=True, random_state=7537) ## random_state not used
        graph = get_dataset_offline(f'{name}_added_failure.dglgraph', f'{root}/', to_homo=True, random_state=7537) ## random_state not used
        print(list(graph.ndata.keys()))

        if nhop is not None:
            labels = graph.ndata['label']
            pos_idx = labels.nonzero(as_tuple=True)[0]

            if nhop > 0:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(nhop)
                input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, pos_idx)

                edgeids = []
                for block in blocks:
                    bg = dgl.block_to_graph(block)
                    edgeids += bg.edata[dgl.EID]
                edgeids = torch.unique(torch.LongTensor(edgeids))
                graph = dgl.edge_subgraph(graph, edgeids)
            elif nhop == 0:
                graph = dgl.node_subgraph(graph, pos_idx)


        n = graph.number_of_nodes()
        graph.ndata['feature'] = graph.ndata['feature'].float()

        ##### Node csv
        node_dict = {
            "nodeId": torch.arange(graph.number_of_nodes(), dtype=int).tolist(),
            "label": graph.ndata['label'].tolist(),
            "failure": graph.ndata['failure_ConsisGNN'].tolist(),
        }
        features = graph.ndata['feature']

        if nnodes is not None:
            node_dict['nodeId'] = node_dict['nodeId'][:nnodes]
            node_dict['label'] = node_dict['label'][:nnodes]
            node_dict['failure'] = node_dict['failure'][:nnodes]
            features = features[:nnodes]

        n, d = features.shape

        for dd in range(d):
            node_dict[f'feature_{dd}'] = features[:, dd]

        print(f'#Nodes: {len(node_dict["nodeId"])}')
        node_df = pd.DataFrame(node_dict, columns=list(node_dict.keys()))
        if nnodes is not None:
            node_df.to_csv(f"{outdir}/{name}_node_{nnodes}.csv", index=False)
        elif nhop is not None:
            node_df.to_csv(f"{outdir}/{name}_node_{nhop}hop.csv", index=False)
        else:
            node_df.to_csv(f"{outdir}/{name}_node.csv", index=False)

        ##### Edges csv
        edge_dict = {
            "edgeId": torch.arange(graph.num_edges(), dtype=int).tolist(),
            "srcnodes": [],
            "dstnodes": [],
            "etype": [],
        }

        for etype in graph.etypes:
            edges = graph.edges(etype=etype)
            srcnodes = edges[0]
            dstnodes = edges[1]

            if nnodes is not None:
                mask1 = srcnodes < nnodes
                mask2 = dstnodes < nnodes
                mask = mask1 * mask2
                srcnodes = srcnodes[mask]
                dstnodes = dstnodes[mask]

            edge_dict['srcnodes'] += srcnodes.tolist()
            edge_dict['dstnodes'] += dstnodes.tolist()
            edge_dict['etype'] += [etype] * len(srcnodes)
        edge_dict['edgeId'] = torch.arange(len(edge_dict['srcnodes']), dtype=int).tolist()

        print(f'#Edges: {len(edge_dict["edgeId"])}')
        edge_df = pd.DataFrame(edge_dict, columns=list(edge_dict.keys()))
        if nnodes is not None:
            edge_df.to_csv(f"{outdir}/{name}_edge_{nnodes}.csv", index=False)
        elif nhop is not None:
            edge_df.to_csv(f"{outdir}/{name}_edge_{nhop}hop.csv", index=False)
        else:
            edge_df.to_csv(f"{outdir}/{name}_edge.csv", index=False)




def dglgraph2edgelist():
    root = "data"
    outdir = f"{root}/offline"
    os.makedirs(outdir, exist_ok=True)


    datanames = ['yelp', 'amazon', 'tfinance', 'merge']

    for name in datanames:
        print(f'Data: {name}')

        dglgraph = load_graphs(os.path.join("data", "offline", f"{name}.dglgraph"))[0][0]
        print(f'#Nodes: {dglgraph.num_nodes()}, #Edges: {dglgraph.num_edges()}')

        with open(f'{outdir}/{name}.edgelist', 'w') as f:
            for etype in dglgraph.etypes:
                edges = dglgraph.edges(etype=etype)
                srcnodes = edges[0]
                dstnodes = edges[1]

                for i in range(len(srcnodes)):
                    u, v = srcnodes[i], dstnodes[i]
                    f.write(f'{u} {v}\n')

        print(f'Edgelist write into {outdir}/{name}.edgelist')


def addEmbedding2DGLGraph():
    root = "data"
    outdir = f"{root}/offline"
    os.makedirs(outdir, exist_ok=True)


    datanames = ['yelp', 'amazon', 'tfinance', 'merge']

    for name in datanames:
        print(f'Data: {name}')

        dglgraph = load_graphs(os.path.join("data", "offline", f"{name}.dglgraph"))[0][0]
        print(f'#Nodes: {dglgraph.num_nodes()}, #Edges: {dglgraph.num_edges()}')
        n, m = dglgraph.num_nodes(), dglgraph.num_edges()

        ### Load deepwalk embedding
        embs = np.zeros((n, 64), dtype=float)
        with open(f'{outdir}/deepwalk/{name}.deepwalk', 'r') as f:
            lines = f.readlines()[1:]
            # assert len(lines) == n
        for line in lines:
            line = line.split(' ')
            nodeid = int(line[0])
            line = line[1:]
            line = [float(x) for x in line]
            embs[nodeid] = line
        embs = torch.FloatTensor(embs)
        dglgraph.ndata['deepwalk'] = embs

        dgl.save_graphs(f"{outdir}/{name}_added_embs.dglgraph", [dglgraph])

    print(f'========= Merge Graphs ===========')


def addPE2DGLGraph():
    root = "data"
    outdir = f"{root}/offline"
    os.makedirs(outdir, exist_ok=True)


    # datanames = ['yelp', 'amazon', 'tfinance', 'merge']
    datanames = ['weibo', 'reddit', 'tolokers', 'yelp', 'questions', 'elliptic',  'dgraphfin', 'tsocial']

    for name in datanames:
        print(f'Data: {name}')

        dglgraph = load_graphs(os.path.join("data", "offline", f"{name}.dglgraph"))[0][0]
        homograph = dgl.to_homogeneous(dglgraph, ndata=None)
        homograph = dgl.remove_self_loop(homograph)
        print(f'#Nodes: {homograph.num_nodes()}, #Edges: {homograph.num_edges()}')
        n, m = homograph.num_nodes(), homograph.num_edges()

        rwse = dgl.random_walk_pe(homograph, 2)
        lapse = dgl.laplacian_pe(homograph, 2)

        dglgraph.ndata['rwse'] = torch.FloatTensor(rwse)
        dglgraph.ndata['lapse'] = torch.FloatTensor(lapse)
        
        dgl.save_graphs(f"{outdir}/{name}_added_pe.dglgraph", [dglgraph])

    print(f'========= Merge Graphs ===========')



if __name__ == "__main__":

    #### Offline data splitting

    # offline_data_split(1, to_homo=False, nseeds=5, fill_mode="zero")

    #### DGLGraph to CSV
    # dglgraph2CSV(nnodes=100)
    # nhops = [0,1,2]
    # for nhop in nhops:
    #     print(f"nhop={nhop}")
    #     dglgraph2CSV(nhop=nhop)

    #### DGLGraph to edgelist
    # dglgraph2edgelist()


    #### Add pre-generated structral embeddings into DGLGraph
    # addEmbedding2DGLGraph()


    #### Add positional encodings into DGLGraph
    addPE2DGLGraph()
    pass

    
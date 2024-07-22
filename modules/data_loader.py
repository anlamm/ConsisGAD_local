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

    else:
        raise
    
    return graph

    
def get_index_loader_test(name: str, batch_size: int, unlabel_ratio: int=1, training_ratio: float=-1,
                             shuffle_train: bool=True, to_homo:bool=False, fill="zero", random_feature=False, verbose=False, load_offline=False, seed=None):
    assert name in ['yelp', 'amazon', 'tfinance', 'tsocial', 'merge'], 'Invalid dataset name'

    if load_offline:
        graph = load_graphs(os.path.join("data", "offline", f"{name}.dglgraph"))[0][0]

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
        
        valid_loader = torch_dataloader(valid_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
        test_loader = torch_dataloader(test_nids, batch_size=2**power, shuffle=False, drop_last=False, num_workers=4)
        labeled_loader = torch_dataloader(labeled_nids, batch_size=batch_size, shuffle=shuffle_train, drop_last=True, num_workers=0)
        unlabeled_loader = torch_dataloader(unlabeled_nids, batch_size=batch_size * unlabel_ratio, shuffle=shuffle_train, drop_last=True, num_workers=0)


        if random_feature:
            # n, d = graph.ndata['feature'].shape
            # new_features = torch.randn(n, 32).float()
            # graph.ndata['feature'] = new_features
            graph.ndata['feature'] = graph.ndata['random_feature']

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
    outdir = f"{root}/offline"
    os.makedirs(outdir, exist_ok=True)

    #### single dataset
    print(f"================ Single dataset =================")
    datanames = ['yelp', 'amazon', 'tfinance']

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





def dglgraph2CSV():
    
    root = "data"
    outdir = f"{root}/CSV"
    os.makedirs(outdir, exist_ok=True)


    
    datanames = ['yelp', 'amazon', 'tfinance']

    for name in datanames:
        print(f"{name}")

        graph = get_dataset(name, f'{root}/', to_homo=False, random_state=7537) ## random_state not used
        n = graph.number_of_nodes()
        graph.ndata['feature'] = graph.ndata['feature'].float()

        ##### Node csv
        node_dict = {
            "nodeId": torch.arange(graph.number_of_nodes(), dtype=int).tolist(),
            "label": graph.ndata['label'].tolist(),
        }

        features = graph.ndata['feature']
        n, d = features.shape

        for dd in range(d):
            node_dict[f'feature_{dd}'] = features[:, dd]

        node_df = pd.DataFrame(node_dict, columns=list(node_dict.keys()))
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

            edge_dict['srcnodes'] += srcnodes.tolist()
            edge_dict['dstnodes'] += dstnodes.tolist()
            edge_dict['etype'] += [etype] * len(srcnodes)

        edge_df = pd.DataFrame(edge_dict, columns=list(edge_dict.keys()))
        edge_df.to_csv(f"{outdir}/{name}_edge.csv", index=False)






    pass





if __name__ == "__main__":

    #### Offline data splitting

    # offline_data_split(1, to_homo=False, nseeds=5, fill_mode="zero")

    #### DGLGraph to CSV
    dglgraph2CSV()

    pass

    
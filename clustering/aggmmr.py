from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import numpy as np
from sklearn.preprocessing import normalize
import torch
import argparse
import random
import setproctitle

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import torch
import torch.nn as nn

import sklearn.preprocessing as preprocess
import os
import shutil

import networkit as nk
from ogb.nodeproppred import DglNodePropPredDataset

from .hierarchical_leiden import _compute_leiden_communities
from .utils import to_df



def aggmmr_cluster(nx_graph, features, nclass, max_cluster_size=20, use_lcc=False, seed=1234):
    from community import community_louvain
    import networkx as nx
    from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=nclass, random_state=seed)
    km.fit(features)
    centers = km.cluster_centers_ ## (n_clusters, n_features)

    graph = nx_graph
    n,m=graph.number_of_nodes(), graph.number_of_edges()
    degs = graph.degree() ## dict(node:degree)
    print(f'#Before: {graph.number_of_nodes()}\t{graph.number_of_edges()}\t{graph}')

    nid = n
    nodelist = list(graph.nodes())
    for center in centers:
        for i in nodelist:
            # dis = np.linalg.norm(features[i]-center) ## euclidean distance
            # weight = np.exp(-dis/2) * degs[i]
            # graph.add_edge(i, nid, weight=weight)
            graph.add_edge(i, nid)
        nid += 1
    print(f'#After: {graph.number_of_nodes()}\t{graph.number_of_edges()}\t{graph}')

    # graph_nk = nk.nxadapter.nx2nk(graph, weightAttr="weight")
    # algo = nk.community.PLM(graph_nk, refine=True, turbo=True, nm="queue", par="balanced", maxIter=32)
    # plmCommunities = nk.community.detectCommunities(graph_nk, algo=algo)
    # preds = plmCommunities.getVector()[:n]

    print(f'max_cluster_size: {max_cluster_size}')
    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=seed,
    )

    for level in node_id_to_community_map.keys():
        for nodeid in range(n, graph.number_of_nodes()):
            node_id_to_community_map[level].pop(nodeid, None)

    return node_id_to_community_map

if __name__ == '__main__':
    if os.path.exists(f'temp'):
        shutil.rmtree(f'temp')
    os.makedirs(f'temp', exist_ok=True)

    # Load the Cora dataset
    dataset = Planetoid(root='data/Planetoid', name='Cora')

    # Accessing the data
    data = dataset[0]  # The dataset contains only one graph

    print(f'Dataset: {dataset}')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {data.num_node_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')

    # Convert to NetworkX graph
    G_nx = to_networkx(data, to_undirected=True)
    features = data.x.numpy()
    nclass = 7
    max_cluster_size = 20

    node_id_to_community_map = aggmmr_cluster(
        G_nx,
        features,
        nclass,
        max_cluster_size=max_cluster_size,
        use_lcc=True,
        seed=0xDEADBEEF,
    )

    print(f'levels: {list(node_id_to_community_map.keys())}')

    for level in node_id_to_community_map.keys():
        nodeid2commid = node_id_to_community_map[level]
        print(f'level {level}: #nodes: {len(nodeid2commid)}')

        with open(f'temp/node2comm_level_{level}.txt', 'w') as f:
            for nodeid, commid in nodeid2commid.items():
                f.write(f'{nodeid}\t{commid}\n')

        commids = np.unique(np.array(list(nodeid2commid.values())))
        with open(f'temp/commsta_level_{level}.txt', 'w') as f:
            exceed_max_size_cnt = 0
            for commid in commids:
                cnt = (np.array(list(nodeid2commid.values())) == commid).sum()
                f.write(f'{commid}\t{cnt}\n')
                if cnt > max_cluster_size:
                    exceed_max_size_cnt += 1
        print(f'@Exceed{max_cluster_size}: {level}\t{exceed_max_size_cnt}')
    df = to_df(node_id_to_community_map=node_id_to_community_map)
    df.to_csv(f'temp/Communities.csv', sep=',', index=False)
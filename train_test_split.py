import dgl
import torch
import numpy as np

amz_graph, _ = dgl.load_graphs('./data/offline/amazon.dglgraph')
print("number of graphs:", len(amz_graph))
for i in range(len(amz_graph)):
    print("graph", i)
    print("number of nodes:", amz_graph[i].num_nodes())
    print("number of edges:", amz_graph[i].num_edges())
    print("Node data:", list(amz_graph[i].ndata.keys()))
    for key in amz_graph[i].ndata.keys():
        print("Node data", key, ":", amz_graph[i].ndata[key].shape)
    print("Edge types:", amz_graph[i].canonical_etypes)
    # print(amz_graph[i].ndata['train_mask'].sum())
    # print(amz_graph[i].ndata['val_mask'].sum())
    # print(amz_graph[i].ndata['test_mask'].sum())

train_pct = 0.6
val_pct = 0.05
test_pct = 0.1

def train_test_split(graph, train_pct, val_pct, test_pct):
    num_nodes = graph.num_nodes()
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(train_pct * num_nodes)
    val_size = int(val_pct * num_nodes)
    test_size = int(test_pct * num_nodes)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask

def train_test_split_by_deg(graph, train_pct, val_pct, test_pct):
    num_nodes = graph.num_nodes()
    in_deg = np.zeros(num_nodes)
    for etype in graph.canonical_etypes:
        in_deg += np.array(graph.in_degrees(etype = etype))
    indices = np.argsort(in_deg)
    # indices = np.arange(num_nodes)
    # np.random.shuffle(indices)
    train_size = int(train_pct * num_nodes)
    val_size = int(val_pct * num_nodes)
    test_size = int(test_pct * num_nodes)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:train_size + val_size + test_size]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask


# amz_graph[0].ndata['train_mask'], amz_graph[0].ndata['val_mask'], amz_graph[0].ndata['test_mask'] = train_test_split(amz_graph[0], train_pct, val_pct, test_pct)
# dgl.save_graphs('./data/offline/amazon_split01.dglgraph', amz_graph)

amz_graph[0].ndata['train_mask'], amz_graph[0].ndata['val_mask'], amz_graph[0].ndata['test_mask'] = train_test_split_by_deg(amz_graph[0], train_pct, val_pct, test_pct)
dgl.save_graphs('./data/offline/amazon_split02.dglgraph', amz_graph)
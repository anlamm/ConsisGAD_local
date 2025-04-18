import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import dgl
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot feature and degree distributions for an Amazon DGL graph.')
parser.add_argument('--graph-name', type=str, required=True, 
                    help='Name of the graph file (e.g., amazon01, amazon02)')
args = parser.parse_args()

# Construct file path for the graph
graph_file = f'./data/offline/{args.graph_name}.dglgraph'
output_dir = './data/offline'
os.makedirs(output_dir, exist_ok=True)

# Load the graph
amz_graph, _ = dgl.load_graphs(graph_file)
graph = amz_graph[0]

# Extract train, validation, and test node indices
train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
val_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=True)[0]
test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=True)[0]

# Print dataset statistics
print(f"Train nodes: {len(train_nids)}")
print(f"Validation nodes: {len(val_nids)}")
print(f"Test nodes: {len(test_nids)}")
etypes = graph.canonical_etypes
print("Edge types:", etypes)

# Print node feature shapes
print("random features: ", graph.ndata['random_feature'].shape)
print("same_features: ", graph.ndata['same_feature'].shape)

# Plot feature distributions
random_feature = graph.ndata['random_feature']
fig, axes = plt.subplots(8, 4, figsize=(20, 40), sharex=False, sharey=False)
axes = axes.flatten()
for feature_idx in range(random_feature.shape[1]):
    train_feature = random_feature[train_nids, feature_idx].numpy()
    val_feature = random_feature[val_nids, feature_idx].numpy()
    test_feature = random_feature[test_nids, feature_idx].numpy()
    axes[feature_idx].hist(train_feature, bins=100, alpha=0.5, label='Train')
    axes[feature_idx].hist(val_feature, bins=100, alpha=0.5, label='Validation')
    axes[feature_idx].hist(test_feature, bins=100, alpha=0.5, label='Test')
    axes[feature_idx].set_title(f'Feature {feature_idx}')
    axes[feature_idx].set_xlabel('Value')
    axes[feature_idx].set_ylabel('Frequency')
    axes[feature_idx].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{args.graph_name}_features.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot in-degree distribution
train_in_deg = np.zeros(train_nids.shape[0])
val_in_deg = np.zeros(val_nids.shape[0])
test_in_deg = np.zeros(test_nids.shape[0])
for etype in etypes:
    in_deg = graph.in_degrees(etype=etype)
    train_in_deg += np.array(in_deg[train_nids])
    val_in_deg += np.array(in_deg[val_nids])
    test_in_deg += np.array(in_deg[test_nids])
plt.figure(figsize=(8, 6))
plt.hist(train_in_deg, bins=100, alpha=0.5, label='Train')
plt.hist(val_in_deg, bins=100, alpha=0.5, label='Validation')
plt.hist(test_in_deg, bins=100, alpha=0.5, label='Test')
plt.xlabel('In-degree')
plt.ylabel('Frequency')
plt.title('In-degree Distribution')
plt.legend()
plt.savefig(os.path.join(output_dir, f'{args.graph_name}_in_degree.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot out-degree distribution
train_out_deg = np.zeros(train_nids.shape[0])
val_out_deg = np.zeros(val_nids.shape[0])
test_out_deg = np.zeros(test_nids.shape[0])
for etype in etypes:
    out_deg = graph.out_degrees(etype=etype)
    train_out_deg += np.array(out_deg[train_nids])
    val_out_deg += np.array(out_deg[val_nids])
    test_out_deg += np.array(out_deg[test_nids])
plt.figure(figsize=(8, 6))
plt.hist(train_out_deg, bins=100, alpha=0.5, label='Train')
plt.hist(val_out_deg, bins=100, alpha=0.5, label='Validation')
plt.hist(test_out_deg, bins=100, alpha=0.5, label='Test')
plt.xlabel('Out-degree')
plt.ylabel('Frequency')
plt.title('Out-degree Distribution')
plt.legend()
plt.savefig(os.path.join(output_dir, f'{args.graph_name}_out_degree.png'), dpi=300, bbox_inches='tight')
plt.close()
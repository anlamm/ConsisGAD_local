# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run and _compute_leiden_communities methods definitions."""

import logging
from typing import Any

import networkx as nx
from .packages.graspologic.partition import hierarchical_leiden

from .utils import stable_largest_connected_component, to_df

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import numpy as np

import os
import shutil

log = logging.getLogger(__name__)


def run(graph: nx.Graph, args: dict[str, Any]) -> dict[int, dict[str, list[str]]]:
    """Run method definition."""
    max_cluster_size = args.get("max_cluster_size", 10)
    use_lcc = args.get("use_lcc", True)
    if args.get("verbose", False):
        log.info(
            "Running leiden with max_cluster_size=%s, lcc=%s", max_cluster_size, use_lcc
        )

    node_id_to_community_map = _compute_leiden_communities(
        graph=graph,
        max_cluster_size=max_cluster_size,
        use_lcc=use_lcc,
        seed=args.get("seed", 0xDEADBEEF),
    )
    levels = args.get("levels")

    # If they don't pass in levels, use them all
    if levels is None:
        levels = sorted(node_id_to_community_map.keys())

    results_by_level: dict[int, dict[str, list[str]]] = {}
    for level in levels:
        result = {}
        results_by_level[level] = result
        for node_id, raw_community_id in node_id_to_community_map[level].items():
            community_id = str(raw_community_id)
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)
    return results_by_level


# Taken from graph_intelligence & adapted
def _compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph,
    max_cluster_size: int,
    use_lcc: bool,
    seed=0xDEADBEEF,
) -> dict[int, dict[str, int]]:
    """Return Leiden root communities."""
    if use_lcc:
        graph = stable_largest_connected_component(graph)

    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

    return results

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
    max_cluster_size = 20

    node_id_to_community_map = _compute_leiden_communities(
        graph=G_nx,
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
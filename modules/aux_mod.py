import torch
import dgl
import dgl.function as fn
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import time
import csv
import math
import modules.mod_utls as m_utls
import random
from modules.conv_mod import CustomLinear
from modules.mr_conv_mod import build_mlp


Tensor = torch.tensor


def fixed_augmentation(graph, seed_nodes, sampler, aug_type: str, p: float=None):
    assert aug_type in ['dropout', 'dropnode', 'dropedge', 'replace', 'drophidden', 'none']
    with graph.local_scope():
        if aug_type == 'dropout':
            input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seed_nodes)
            blocks[0].srcdata['feature'] = F.dropout(blocks[0].srcdata['feature'], p)
            
        elif aug_type == 'dropnode':
            input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seed_nodes)
            blocks[0].srcdata['feature'] = m_utls.drop_node(blocks[0].srcdata['feature'], p)
            
        elif aug_type == 'dropedge':
            del_edges = {}
            for et in graph.etypes:
                _, _, eid = graph.in_edges(seed_nodes, etype=et, form='all')
                num_remove = math.floor(eid.shape[0] * p)
                del_edges[et] = eid[torch.randperm(eid.shape[0])][:num_remove]
            aug_graph = graph
            for et in del_edges.keys():
                aug_graph = dgl.remove_edges(aug_graph, del_edges[et], etype=et)
            input_nodes, output_nodes, blocks = sampler.sample_blocks(aug_graph, seed_nodes)
            
        elif aug_type == 'replace':
            raise Exception("The Replace sample is not implemented!")
            
        elif aug_type == 'drophidden':
            input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seed_nodes)
            
        else:
            input_nodes, output_nodes, blocks = sampler.sample_blocks(graph, seed_nodes)
            
        return input_nodes, output_nodes, blocks

class RandomWalkSampler(dgl.dataloading.Sampler):
    def __init__(self, fanout : int, walk_length: int):
        super().__init__() 
        self.fanout = fanout  ### num of walk for each node
        self.walk_length = walk_length ### length of walk for each node

    def sample(self, g, seed_nodes):
        #### g: homogeneous graph
        seed_nodes = torch.LongTensor(seed_nodes)
        seed_nodes = seed_nodes.tile(self.fanout)
        rw, _ = dgl.sampling.random_walk(g, seed_nodes, length=self.walk_length)
        

        srcnodes = rw[:, -1]
        dstnodes = rw[:, 0]

        subg = dgl.graph((srcnodes, dstnodes))
        n, m = subg.number_of_nodes(), subg.number_of_edges()
        d = g.ndata['feature'].shape[1]
        subg.ndata['feature'] = torch.zeros(n, d)
        subg.ndata['feature'][srcnodes] = g.ndata['feature'][srcnodes]
        subg.ndata['feature'][dstnodes] = g.ndata['feature'][dstnodes]

        subg = dgl.to_block(subg, seed_nodes)
        return subg
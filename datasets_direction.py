#!/usr/bin/env python
# coding: utf-8

import torch
import pickle
import os
import ipdb
import numpy as np
import pandas as pd

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from torch_sparse import coalesce
# from randomperm_code import random_planetoid_splits
from sklearn.feature_extraction.text import CountVectorizer

from data_utils import load_citation_dataset #, load_LE_dataset, \
#     load_yelp_dataset, load_cornell_dataset
from data_utils_direction import load_citation_dataset_direction, load_other_directed_graph, \
load_citation_pubmed_dataset_direction, load_synthetic_dataset, load_mail_dataset_direction
from torch_geometric_signed_directed.utils import  in_out_degree
from torch_geometric_signed_directed.data import load_directed_real_data

class AddHypergraphSelfLoops(torch_geometric.transforms.BaseTransform):
    def __init__(self, ignore_repeat=True):
        super().__init__()
        # whether to detect existing self loops
        self.ignore_repeat = ignore_repeat
    
    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_hyperedges = data.num_hyperedges

        node_added = torch.arange(num_nodes, device=edge_index.device, dtype=torch.int64)
        if self.ignore_repeat:
            # 1. compute hyperedge degree
            hyperedge_deg = torch.zeros(num_hyperedges, device=edge_index.device, dtype=torch.int64)
            hyperedge_deg = hyperedge_deg.scatter_add(0, edge_index[1], torch.ones_like(edge_index[1]))
            hyperedge_deg = hyperedge_deg[edge_index[1]]

            # 2. if a node has a hyperedge with degree 1, then this node already has a self-loop
            has_self_loop = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.int64)
            has_self_loop = has_self_loop.scatter_add(0, edge_index[0], (hyperedge_deg == 1).long())
            node_added = node_added[has_self_loop == 0]

        # 3. create dummy hyperedges for other nodes who have no self-loop
        hyperedge_added = torch.arange(num_hyperedges, num_hyperedges + node_added.shape[0])
        edge_indx_added = torch.stack([node_added, hyperedge_added], 0)
        edge_index = torch.cat([edge_index, edge_indx_added], -1)

        # 4. sort along w.r.t. nodes
        _, sorted_idx = torch.sort(edge_index[0])
        data.edge_index = edge_index[:, sorted_idx].long()

        return data

class HypergraphDataset(InMemoryDataset):

    cocitation_list = ['cora', 'citeseer', 'pubmed']
    cornell_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'congress-bills', 'senate-committees'] + \
        ['synthetic-0.1', 'synthetic-0.15', 'synthetic-0.2', 'synthetic-0.3', 'synthetic-0.35', 'synthetic-0.4', 'synthetic-0.5']
    mail = ['enroll', 'EU']
    synthetic = ['node:500_subset-min=3_subset-max=10_N-intra=30_N-inter=10'] + \
    ['node:500_subset-min=3_subset-max=10_N-intra=30_N-inter=20', 'node:500_subset-min=3_subset-max=10_N-intra=30_N-inter=30', 'node:500_subset-min=3_subset-max=10_N-intra=30_N-inter=40'] +\
    ['node:500_subset-min=3_subset-max=10_N-intra=30_N-inter=50', 'node:500_subset-min=3_subset-max=10_N-intra=50_N-inter=20', 'node:500_subset-min=3_subset-max=10_N-intra=70_N-inter=20'] + \
    ['node:500_subset-min=3_subset-max=10_N-intra=90_N-inter=20', 'node:500_subset-min=3_subset-max=10_N-intra=70_N-inter=5', 'node:500_subset-min=3_subset-max=10_N-intra=90_N-inter=5'] + \
    ['node:500_subset-min=3_subset-max=10_N-intra=110_N-inter=5' ]   
    #['synthetic-n_vertex:500-noise_level:0.1-change_prob:0.3-undirected:True-big_class:True', 'synthetic-n_vertex:500-noise_level:0.0-change_prob:0.4-undirected:True-big_class:True'] +\
    
    
    
    existing_dataset = cocitation_list + cornell_list
    
    @staticmethod
    def parse_dataset_name(name):
        name_cornell = '-'.join(name.split('-')[:-1])
        extras = {}
        if name_cornell in HypergraphDataset.cornell_list:
            extras['feature_dim'] = int(name.split('-')[-1])
            name = name_cornell

        return name, extras

    @staticmethod
    def dataset_exists(name):
        name, _ = HypergraphDataset.parse_dataset_name(name)
        return (name in HypergraphDataset.existing_dataset)

    def __init__(self, root, name,  path_to_download='./raw_data',
        feature_noise = None, transform = None, pre_transform = None, second_name = None):
        # features noise is only for cornell --> non ci interessa MAI
        #assert self.dataset_exists(name), f'Dataset {name} is not defined'
        self.name = name
        self.feature_noise = feature_noise
        self.path_to_download = path_to_download
        self.root = root
        self.second_name = second_name

        if not os.path.isdir(root):
            os.makedirs(root)

        # 1. this line will sequentially call download, preprocess, and save data
        super(HypergraphDataset, self).__init__(root, transform, pre_transform)
        # 2. load preprocessed data
        self.data, self.slices = torch.load(self.processed_paths[0])
        if (name in self.cocitation_list) or (name in self.mail):
            # 2. extract to V->E edges
            edge_index = self.data.edge_index

            # sort to [V,E] (increasing along edge_index[0])
            _, sorted_idx = torch.sort(edge_index[0])
            edge_index = edge_index[:, sorted_idx].long()
            num_nodes, num_hyperedges = self.data.num_nodes, self.data.num_hyperedges
            assert ((num_nodes + num_hyperedges - 1) == self.data.edge_index.max().item())

            # search for the first E->V edge, as we assume the source node is sorted like [V | E]
            cidx = torch.where(edge_index[0] == num_nodes)[0].min()
            self.data.edge_index = edge_index[:, :cidx].long() # the first vertex start from 1 instead of 0
            # sorting degli weight edges
            self.data.edge_weight = self.data.edge_weight[sorted_idx][:len(self.data.edge_index[0])]
            # reindex the hyperedge starting from zero
            self.data.edge_index[1] -= (num_nodes) # the first incident column matrix start from 0

            # 3. extract to V->E edges
            self.data.edge_index = self.data.edge_index.long()
            if (not torch.is_tensor(self.data.x)) & (not self.data.x is None):
                self.data.x = torch.from_numpy(self.data.x)
        elif name in self.synthetic:
            # 2. extract to V->E edges
            edge_index = self.data.edge_index

            # sort to [V,E] (increasing along edge_index[0])
            _, sorted_idx = torch.sort(edge_index[0])
            edge_index = edge_index[:, sorted_idx].long()
            num_nodes, num_hyperedges = self.data.num_nodes, self.data.num_hyperedges
            assert ((num_nodes + num_hyperedges - 1) == self.data.edge_index.max().item())

            # search for the first E->V edge, as we assume the source node is sorted like [V | E]
            cidx = torch.where(edge_index[0] == num_nodes)[0].min()
            self.data.edge_index = edge_index[:, :cidx].long() #-1 # the first vertex start from 1 instead of 0
            # sorting degli weight edges
            self.data.edge_weight = self.data.edge_weight[sorted_idx][:len(self.data.edge_index[0])]
            # reindex the hyperedge starting from zero
            self.data.edge_index[1] -= num_nodes #(num_nodes-1) # the first incident column matrix start from 0
            # 3. extract to V->E edges
            self.data.edge_index = self.data.edge_index.long()

            



    @property
    def raw_dir(self):
        return os.path.join(self.root, './raw_data')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'{self.name}_noise_{self.feature_noise}']
        else:
            file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'data_noise_{self.feature_noise}.pt']
        else:
            file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features

    @property
    def num_classes(self):
        return self.data.num_classes

    @staticmethod
    def save_data_to_pickle(data, save_dir, file_name):
        '''
        if file name not specified, use time stamp.
        '''
        file_path = os.path.join(save_dir, file_name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(file_path, 'bw') as f:
            pickle.dump(data, f)
        return file_path

    def download(self):
        for file_name in self.raw_file_names:
            path_raw_file = os.path.join(self.raw_dir, file_name)
            if os.path.isfile(path_raw_file):
                continue
            if not os.path.isdir(self.path_to_download):
                raise ValueError(f'Path to downloaded hypergraph dataset does not exist!', self.path_to_download)

            dataset_name, extra = self.parse_dataset_name(self.name)

            # file not exist, so we create it and save it there.
            if dataset_name in self.cocitation_list:
                if dataset_name == 'pubmed':
                    raw_data = load_citation_pubmed_dataset_direction(path = self.path_to_download, dataset = dataset_name)
                else:
                    raw_data = load_citation_dataset_direction(path = self.path_to_download, dataset = dataset_name)
                #raw_data = load_citation_dataset(path = self.path_to_download, dataset = dataset_name)
            elif dataset_name in self.mail:
                raw_data = load_mail_dataset_direction(path = self.path_to_download, dataset = dataset_name)
            elif dataset_name in self.synthetic:
                raw_data = load_synthetic_dataset(path = self.path_to_download, dataset = dataset_name)
            else:
                if dataset_name.lower() in ['wikics', 'telegram']:
                    self.path_to_download = os.path.join(self.path_to_download, self.second_name)
                data = load_directed_real_data(dataset=dataset_name, root=self.path_to_download, name=self.second_name)
                raw_data = load_other_directed_graph(data)

            
            self.save_data_to_pickle(raw_data, save_dir = self.raw_dir, file_name = file_name)

    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f)
        raw_data = raw_data if self.pre_transform is None else self.pre_transform(raw_data)
        torch.save(self.collate([raw_data]), self.processed_paths[0])

    def __repr__(self):
        return '{}(feature_noise={})'.format(self.name, self.feature_noise)

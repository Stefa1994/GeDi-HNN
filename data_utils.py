

import torch
import pickle
import os
import ipdb
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from torch_sparse import coalesce
from sklearn.feature_extraction.text import CountVectorizer


def count_element(my_dict):
    # Count the total number of elements in the values of the dictionary
    total_elements = sum(len(v) for v in my_dict.values())

# Count the number of values in the dictionary
    num_values = len(my_dict)

# Calculate the average number of elements per value
    average_elements = total_elements / num_values if num_values != 0 else 0

    print(f'The average number of elements per value is: {average_elements}')

def load_LE_dataset(path, dataset):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))
    
    file_name = f'{dataset}.content'
    p2idx_features_labels = os.path.join(path, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-1])
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))


    print ('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    file_name = f'{dataset}.edges'
    p2edges_unordered = os.path.join(path, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print ('load edges')


    projected_features = torch.FloatTensor(np.array(features.todense()))

    
    # From adjacency matrix to edge_list
    edge_index = edges.T 
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1
    
    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1
    
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    
    # build torch data class
    data = Data(x = torch.FloatTensor(np.array(features[:num_nodes].todense())), 
            edge_index = torch.LongTensor(edge_index),
            y = labels[:num_nodes])

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    
    data.num_features = data.x.shape[-1]
    data.num_classes = len(np.unique(labels[:num_nodes].numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = num_he
    
    return data

# from direcetd graph to undirecetd graph
def load_mail(path, dataset):
    # then load node labels:
#    with open(os.path.join(path, dataset, 'unweight_graph.pickle'), 'rb') as f:
    with open(os.path.join(path, dataset, 'weight_graph.pickle'), 'rb') as f:
        graph = pickle.load(f)
       
    
    graph = np.array(graph)

    hypergraph, copy = {}, {}
    for citation in graph:
        s, d = citation[1], citation[0]
        if s not in hypergraph.keys(): 
            hypergraph[s], copy[s] = set(), set()
        hypergraph[s].add(d)
        copy[s].add(d)

    for k in hypergraph.keys():
        if len(hypergraph[k]) < 3: copy.pop(k)
    hypergraph = copy
    # then load node labels:
    #with open(os.path.join(path, dataset,  'unweight_label.pickle'), 'rb') as f:
    with open(os.path.join(path, dataset,  'weight_label.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes = len(labels)
    print(f'number of nodes:{num_nodes}')
    

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int64)
    edge_index = torch.LongTensor(edge_index)
    labels = torch.LongTensor(labels)
    data = Data(edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = len(hypergraph)
   
    
    return data


# from direcetd graph to undirecetd graph
def load_other_graph(dataset):

    graph = np.array(dataset.edge_index).T
    hypergraph, copy = {}, {}
    for citation in graph:
        s, d = citation[1], citation[0]
        if s not in hypergraph.keys(): 
            hypergraph[s], copy[s] = set(), set()
        hypergraph[s].add(d)
        copy[s].add(d)

    for k in hypergraph.keys():
        if len(hypergraph[k]) < 3: copy.pop(k)
    hypergraph = copy

    num_nodes, feature_dim = dataset.x.shape
    assert num_nodes == len(dataset.y)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')
    
    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int64)
    edge_index = torch.LongTensor(edge_index)

    dataset.edge_index = edge_index

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    dataset.edge_index, dataset.edge_attr = coalesce(dataset.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
    dataset.num_features = dataset.x.shape[-1]
    dataset.num_classes = len(np.unique(dataset.y.numpy()))
    dataset.num_nodes = num_nodes
    dataset.num_hyperedges = len(hypergraph)

    
    return dataset


def load_citation_dataset(path, dataset):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

    # first load node features:
    with open(os.path.join(path, dataset,  'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(os.path.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(os.path.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int64)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = len(hypergraph)

    return data

def load_synthetic_dataset(path, dataset, directed=False):

    '''
    Dataset loading for syntehtic dataset
    '''

    print(f'Loading Synthetic hypergraph dataset')

    # then load node labels:
    with open(os.path.join(path, dataset, 'label.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes = len(labels)
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}')
    #labels = torch.LongTensor([int(x) -1 for x in labels])
    labels = torch.LongTensor([int(x) for x in labels])

    with open(os.path.join(path, dataset, 'hypergraph_undirected.pickle'), 'rb') as f:

        hypergraph = pickle.load(f)
 
    edge_idx = num_nodes
    node_list = []
    edge_list = []
    num_hyperedges = 0
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)
        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size
        edge_idx += 1
        num_hyperedges += 1
     
            
    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int64)
    edge_index = torch.LongTensor(edge_index)
    data = Data(edge_index = edge_index,
                y = labels)
    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
    data.num_nodes = num_nodes
    data.num_hyperedges = num_hyperedges
    data.num_classes = len(np.unique(labels.numpy()))
    return data


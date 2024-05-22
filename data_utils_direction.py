import torch
import pickle
import os
import ipdb
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from torch_sparse import coalesce
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx

def load_mail_dataset_direction(path, dataset):
    # then load node labels:
    with open(os.path.join(path, dataset, 'weight_graph.pickle'), 'rb') as f:
        graph = pickle.load(f)
       
    
    graph = np.array(graph)

    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        ste, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(ste)
        copy[t].add(ste)

    
    # then load node labels:
    with open(os.path.join(path, dataset,  'weight_label.pickle'), 'rb') as f:
        labels = pickle.load(f)
    num_nodes = len(labels)
    print(f'number of nodes:{num_nodes}')
    labels = torch.LongTensor(labels)

    data = processing_without_feature(hypergraph=hypergraph, labels=labels)
    return data

# from direcetd graph to undirecetd graph
def load_other_directed_graph(dataset):
    graph = np.array(dataset.edge_index).T

    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        i, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(i)
        copy[t].add(i)


        
    num_nodes, feature_dim = dataset.x.shape
    assert num_nodes == len(dataset.y)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

   
    data = processing2(hypergraph=hypergraph, data=dataset)
    
    return data

def load_synthetic_dataset(path, dataset):
   
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

    labels = torch.LongTensor([int(x) for x in labels]) #torch.LongTensor([int(x) -1 for x in labels])

    # The last, load hypergraph.
    with open(os.path.join(path, dataset, 'hypergraph_directed.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')


    edge_idx = num_nodes
    node_list = []
    edge_list = []
    num_hyperedges = 0
    edge_weight = []
    # Handling both undirecetd and directed hyperedge!
    for k, v in hypergraph.items():
        if v == [()]:
            cur_he = k
            cur_size = len(cur_he)
            node_list += list(cur_he)
            edge_list += [edge_idx] * cur_size
            edge_idx += 1
            edge_weight += [1] * len(list(cur_he))
            num_hyperedges += 1
        else:
            cur_he1 = list(k)
            cur_he2 = v
            cur_he = cur_he1 + list(cur_he2)
            cur_size = len(cur_he)
            node_list += cur_he
            edge_list += [edge_idx] * cur_size
            edge_idx += 1
            edge_weight += [1j]* len(list(cur_he1)) + [1] * len(list(cur_he2))
            num_hyperedges += 1


# double the weights
    edge_weight = edge_weight * 2 
    edge_index = np.array([node_list + edge_list,
                edge_list + node_list], dtype = np.int64)
    
    edge_index = torch.LongTensor(edge_index)
    edge_weight = torch.tensor(np.array(edge_weight, dtype=np.complex_), dtype=torch.cfloat)
    data = Data(edge_index = edge_index, 
                edge_weight = edge_weight,
                y = labels)
    total_num_node_id_he_id = edge_index.max() + 1      
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
    None, 
    total_num_node_id_he_id, 
    total_num_node_id_he_id)
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = num_hyperedges
    return data    


def load_citation_dataset_direction(path, dataset):
    
    file_name_content = f'{dataset}.content'
    p2idx_features_labels = os.path.join(path, dataset,  file_name_content)
    content = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    
    # read citation graph
    file_name = f'{dataset}.cites'
    p2idx_citation_labels = os.path.join(path, dataset, file_name)
    with open(p2idx_citation_labels, "r") as f: 
        cites = f.readlines()
    indices = {j: i for i, j in enumerate(content[:, 0])}
    citations, n = [], 0

    for c in cites:
        c = c.strip("\n").split("\t")
        if c[0] in indices.keys() and c[1] in indices.keys():
            citations.append([c[0], c[1]])
            n = n + 1
    citations = np.array(citations)
    graph = np.array(list(map(indices.get, citations.flatten())), dtype=np.int32).reshape(citations.shape)


    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        i, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(i)
        copy[t].add(i)




  
        
        # first load node features:
    with open(os.path.join(path, dataset,  'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(os.path.join(path, dataset,  'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
 

    data = processing(hypergraph=hypergraph, labels=labels, features=features)
    return data

def processing(hypergraph, labels, features):
        num_nodes = len(labels)
        edge_idx = num_nodes
        node_list = []
        edge_list = []
        edge_weight = []
        for k, v in hypergraph.items():
            cur_he1 = [k]
            cur_he2 = v
            cur_he = list(cur_he1) + list(cur_he2)
            cur_size = len(cur_he)
            node_list += list(cur_he)
            edge_list += [edge_idx] * cur_size
            edge_idx += 1
            edge_weight += [1j]* len(list(cur_he1)) + [1] * len(list(cur_he2))

        # double the weights
        edge_weight = edge_weight * 2 

        edge_index = np.array([node_list + edge_list,
                    edge_list + node_list], dtype = np.int64)
        

        edge_index = torch.LongTensor(edge_index)
        edge_weight = torch.tensor(np.array(edge_weight, dtype=np.complex_), dtype=torch.cfloat)
        data = Data(x = features,
                    edge_index = edge_index, 
                    edge_weight = edge_weight,
                    y = labels)

        total_num_node_id_he_id = edge_index.max() + 1      
        data.edge_index, data.edge_attr = coalesce(data.edge_index, 
        None, 
        total_num_node_id_he_id, 
        total_num_node_id_he_id)
        data.num_classes = len(np.unique(labels.numpy()))
        data.num_nodes = num_nodes
        data.num_hyperedges = len(hypergraph)
        return data

def processing_without_feature(hypergraph, labels):
        num_nodes = len(labels)
        edge_idx = num_nodes
        node_list = []
        edge_list = []
        edge_weight = []
        for k, v in hypergraph.items():
            cur_he1 = [k]
            cur_he2 = v
            cur_he = list(cur_he1) + list(cur_he2)
            cur_size = len(cur_he)
            node_list += list(cur_he)
            edge_list += [edge_idx] * cur_size
            edge_idx += 1
            edge_weight += [1j]* len(list(cur_he1)) + [1] * len(list(cur_he2))

        # double the weights
        edge_weight = edge_weight * 2 

        edge_index = np.array([node_list + edge_list,
                    edge_list + node_list], dtype = np.int64)
        

        edge_index = torch.LongTensor(edge_index)
        edge_weight = torch.tensor(np.array(edge_weight, dtype=np.complex_), dtype=torch.cfloat)
        data = Data(edge_index = edge_index, 
                    edge_weight = edge_weight,
                    y = labels)

        total_num_node_id_he_id = edge_index.max() + 1      
        data.edge_index, data.edge_attr = coalesce(data.edge_index, 
        None, 
        total_num_node_id_he_id, 
        total_num_node_id_he_id)
        data.num_classes = len(np.unique(labels.numpy()))
        data.num_nodes = num_nodes
        data.num_hyperedges = len(hypergraph)
        return data

def processing2(hypergraph, data):
        n =  len(data.y)
        d = len(hypergraph)
    
        # Initialize a matrix of zeros
        #H = np.zeros((n, d), dtype=np.complex128)
        values = []
        rows = []
        cols = []
        # Populate the matrix based on the dictionary
        print('sono pronto a creare il grafo')
        for i, (key, subdict) in enumerate(hypergraph.items()):
        
            rows.append(key)
            cols.append(i)
            values.append(1j)
            for raw in subdict:
                if raw!= key:
                    rows.append(raw)
                    cols.append(i)
                    values.append(1)
    

        H = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(values), torch.Size([n, d])).coalesce()
      
        
        edge_index = torch.LongTensor(H.indices())
        edge_weight = torch.tensor(np.array(H.values(), dtype=np.complex_), dtype=torch.cfloat)

        data.edge_index = edge_index
        data.edge_weight = edge_weight
        total_num_node_id_he_id = edge_index.max() + 1      
        data.edge_index, data.edge_attr = coalesce(data.edge_index, 
        None, 
        total_num_node_id_he_id, 
        total_num_node_id_he_id)
        data.num_classes = len(np.unique(data.y.numpy()))
        data.num_nodes = n
        data.num_hyperedges = len(hypergraph)
        return data

def load_citation_pubmed_dataset_direction_reverse(path, dataset):
    
    # then load node labels:
    with open(os.path.join(path, dataset,  'hypergraph.pickle'), 'rb') as f:
        hypergraph = pickle.load(f)
    # Create an array of indices for each element in hypergraph2
    reverse_graph = []

    for s, neighbors in hypergraph.items():
        for c in neighbors:
            reverse_graph.append([c, s])
    
    
    graph = np.array(reverse_graph)#- 1
    #graph = np.array(list(map(node_map.get, citations.flatten())), dtype=np.int32).reshape(citations.shape)
    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        ste, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(ste)
        copy[t].add(ste)


    
    #    # first load node features:
    with open(os.path.join(path, dataset,  'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(os.path.join(path, dataset,  'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    data = processing(hypergraph=hypergraph, labels=labels, features=features)
    return data

def load_citation_pubmed_dataset_direction(path, dataset):
    
    file_name_content = 'Pubmed-Diabetes.NODE.paper.tab'
    p2idx_features_labels = os.path.join(path, dataset,  file_name_content)
    num_feats = 500
    num_nodes = 19717 
    feat_data = np.zeros((num_nodes, num_feats))
    labels_2 = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open(p2idx_features_labels) as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels_2[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])



     # read citation graph
    file_name = 'Pubmed-Diabetes.DIRECTED.cites.tab'
    p2idx_citation_labels = os.path.join(path, dataset, file_name)
    citations, n = [], 0
    with open(p2idx_citation_labels, "r") as f: 
        cites = f.readlines()
    for c in cites:
        try:
            c = c.strip().split("\t")
            paper1 = node_map[c[1].split(":")[1]]
            paper2 = node_map[c[-1].split(":")[1]]
            citations.append([paper1, paper2])
            n = n + 1
        except:
            continue
    graph = np.array(citations)
    
    hypergraph, copy = {}, {}

    for citation in tqdm(graph):
        ste, t = citation[1], citation[0] # Correct direction of links!
        if t not in hypergraph.keys():
            hypergraph[t], copy[t] = set(), set()
            hypergraph[t].add(t)

        hypergraph[t].add(ste)
        copy[t].add(ste)
    


    # first load node features:
    with open(os.path.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()
    # then load node labels:
    with open(os.path.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)
    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')
    features = torch.FloatTensor(features)
    #labels = torch.LongTensor(labels)
    labels = torch.LongTensor(labels_2.flatten())
    
    data = processing(hypergraph=hypergraph, labels=labels, features=features)
    return data



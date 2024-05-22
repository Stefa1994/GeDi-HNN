import torch.nn as nn
import torch
import torch as th
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dgl.mock_sparse import create_from_coo, diag, identity

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat




def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, args=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H_sparse(H, args=args)  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, args))
        return G


def _generate_G_from_H_sparse(H, add_self_loop=False, sigma=None, args=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    # add_self_loop = False
    if args is not None:
        sigma = args.sigma
    else:
        sigma = -1

    #H = coo_matrix(H)
    n_edge = H.shape[1]  # 4024
    # the weight of the hyperedge
    W = np.ones(n_edge)

    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # [4024]


    DE = DE.tolist()[0]
    invDE = np.power(DE, sigma)
    invDE[np.isinf(invDE)] = 0
    invDE = coo_matrix((invDE, (range(n_edge), range(n_edge))), shape=(n_edge, n_edge))
    K = H * invDE * H.T
    # if args.add_self_loop:
    print('renormalization!!')
    K += sp.eye(H.shape[0])

    DV = np.sum(K, 0).tolist()[0]
    invDV = np.power(DV, -0.5)
    invDV[np.isinf(invDV)] = 0
    DV2 = coo_matrix((invDV, (range(H.shape[0]), range(H.shape[0]))), shape=(H.shape[0], H.shape[0]))


    G = DV2 * K * DV2

    return G




def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, gamma=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param gamma: prob
    :return: N_object X N_hyperedge
    """

    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))

    for center_idx in range(n_obj):
        # dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(-dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (gamma * avg_dis ** 2))
            else:
                H[node_idx, center_idx] = 1.0


    return H


def cal_distance_map(X):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param gamma: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    dis_mat = -Eu_dis(X)

    return dis_mat


def construct_H_with_KNN(dis_mat, K_neigs, split_diff_scale=False, is_probH=True, gamma=1):
    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, gamma)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False, incidence_v=100, incidence_e=50,
                 init_dist=None, args=None):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        self.args = args
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features
        self.lam4=args.lam4
        if self.lam4!=0:
            print("lam4 is not zero!!!!!!! wrong")
            exit(0)
        self.lam0=args.lam0
        self.lam1=args.lam1
        self.alpha=args.alp if args.alp !=0 else 1/(1+args.lam4+args.lam0+args.lam1)
        self.num_steps=args.prop_step
        self.out_features = out_features
        self.residual = residual
        self.notresidual=args.notresidual
        self.twoHgamma=args.twoHgamma
        self.adj = None
        self.normalize_type=args.normalize_type
        if args.H:
            H = {}
            for t in ["beta","gamma1","gamma2"]:

                if args.notresidual:
                    H[t] = torch.rand(in_features, in_features)
                    bound = 4/in_features # normal
                    nn.init.normal_(H[t], 0, bound)
                    H[t] = nn.Parameter(H[t])
                else:


                    H[t] = torch.rand(in_features, in_features)
                    bound =1/in_features # normal
                    nn.init.normal_(H[t], 0, bound)
                    H[t] = H[t] + th.eye(in_features)
                
                
                    H[t] = nn.Parameter(H[t])
                
            self.H = nn.ParameterDict(H)

        else:
            self.H=None
       

        self.init_attn=None
        self.reset_parameters()

    def reset_parameters(self):
        pass
    
    def forward(self, X, A, D):
        A_beta,A_gamma=A
        D_beta,D_gamma,I=D
        ##B is the incidence matrix with N x E
        
        ##after linear and dropout
         # Compute Y = Y0 = f(X; W) using a two-layer MLP.
        H=self.H 
        # Y = Y0 = self.act_fn(self.mlp(X))
        Y = Y0 = X
       

        ####

        # Compute diagonal matrix Q_tild.
        if H is not None:
            # D_stinv=diag((B ).sum(0))
            # # import ipdb
            # # ipdb.set_trace()
            # B_=B @ (D_stinv)**(-1)
            # # Q_tild=self.lam4*L_alpha + self.lam0*D_beta+self.lam1*B_@D_stinv@B_.T + I
            # Q_tild=self.lam1*B_@D_stinv@B_.T + I
            # D_st=diag((B ).sum(1))
            ###############################diagD
            Q_tild= self.lam0*D_beta+self.lam1*D_gamma + I
            diagD=True

            L_gamma=D_gamma.as_sparse()-A_gamma
            # D_st=diag(B.sum(1))
            H_1=H["beta"]
            H_2=H["gamma1"]
            H_3=H["gamma2"]
        else:

            Q_tild= self.lam0*D_beta+self.lam1*D_gamma + I

        # Iteratively compute new Y by equation (6) in the paper.
        for k in range(self.num_steps):
            if H is not None:
                
                # Y_hat = self.lam0 * A_beta @ Y + Y0 + self.lam1 * ( B @B_.T @ Y @ H.T+ B_ @ B.T @ Y @ H- D_st @ Y @ H @ H.T )
                ##diagD
                if diagD:
                    if self.twoHgamma:
                        Y_hat = self.lam0 * (A_beta @ Y @ (H_1+H_1.T)- D_beta @ Y @ H_1 @ H_1.T ) + Y0 + self.lam1/2 * ( L_gamma @ Y + A_gamma @ Y @ (H_2+H_2.T)- D_gamma @ Y @ H_2 @ H_2.T + A_gamma @ Y @ (H_3+H_3.T)- A_gamma @ Y @ H_3 @ H_3.T)
                    else:
                        if self.args.HisI:
                            Y_hat = self.lam0 * (2*A_beta @ Y- D_beta @ Y ) + Y0 + self.lam1 *  A_gamma @ Y 
                        else:

                            Y_hat = self.lam0 * (A_beta @ Y @ (H_1+H_1.T)- D_beta @ Y @ H_1 @ H_1.T ) + Y0 + self.lam1 * ( L_gamma @ Y + A_gamma @ Y @ (H_2+H_2.T)- D_gamma @ Y @ H_2 @ H_2.T )
            else:

                Y_hat = self.lam0 * A_beta @ Y + Y0 + self.lam1 *  A_gamma @ Y 
            Y = (1 - self.alpha) * Y + self.alpha * (Q_tild ** -1) @ Y_hat


        # we have linear out of this module
        return Y

class phenomnn(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, incidence_v=100, incidence_e=50,
                 init_dist=None, args=None):
        super(phenomnn, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(1):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant,args=args))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.in_features = nfeat
        self.out_features = nclass
        self.hiddendim = nhidden
        self.nhiddenlayer = nlayers

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, input, adj, D):
        _layers = []
        x = F.dropout(input, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        # layer_inner = input
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, A=adj, D=D))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        self.adj = con.adj  
        return layer_inner 

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s lamda=%s alpha=%s (%d - [%d:%d] > %d)" % (self.__class__.__name__,self.lamda,
                                                    self.alpha,
                                                    self.in_features,
                                                    self.hiddendim,
                                                    self.nhiddenlayer,
                                                    self.out_features)
    

class GCNModel(nn.Module):
    """
       The model architecture likes:
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="phenomnn",
                 inputlayer=None,
                 outputlayer=None,
                 nbaselayer=0,
                 args=None,
                 ):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "phenomnn", "phenomnn_s".
        :param nbaselayer: the number of layers in one hidden block.
        """
        super(GCNModel, self).__init__()
        self.dropout = dropout
        self.baseblock = baseblock.lower()
        self.nbaselayer = nbaselayer
        self.args = args


        if self.baseblock == "phenomnn":
            self.BASEBLOCK = phenomnn
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))

        self.midlayer = nn.ModuleList()
        for i in range(nhidlayer):
            
            if baseblock.lower() in ['phenomnn']:
                gcb = self.BASEBLOCK(nfeat=nfeat,
                                     nlayers=nbaselayer,
                                     nhidden=nhid,
                                     nclass=nclass,
                                     dropout=dropout,
                                     lamda=args.lamda,
                                     alpha=args.alpha,
                                     variant=args.variant,
                                     args=args,
                                     )

            else:  # gcn
                NotImplementedError("Current baseblock %s is not supported." % (baseblock))
            self.midlayer.append(gcb)
        if baseblock.lower() in ['phenomnn']:
            # self.ingc = nn.Linear(nfeat, nhid)
            # self.outgc = nn.Linear(nhid, nclass)
            # self.fcs = nn.ModuleList([self.ingc, self.outgc])
            self.params1 = self.midlayer[0].params1
            self.params2 = self.midlayer[0].params2
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, data):
        fea = data.x
        adj = data.H
        G = data.G if not None else None
        if self.baseblock=="phenomnn":
            out = self.midlayer[0](input=fea, adj=adj, D=G)
            return out
    
    def data_creation(data, args):
        edge_weight = torch.ones(data.edge_index.size(1),device=data.edge_index.device)
        row, col = data.edge_index.cpu()
        size_col = max(col) + 1
        H = coo_matrix((edge_weight.cpu(), (row, col)), shape=(data.num_nodes, size_col))
        G = _generate_G_from_H_sparse(H, args=args)
        G = sparse_mx_to_torch_sparse_tensor(G)


        H = sparse_mx_to_torch_sparse_tensor(sp.lil_matrix(H))

        src,dst=H.coalesce().indices().to(device)
    
        data.B=create_from_coo(src,dst,torch.ones_like(src).to(torch.float32),shape=H.shape)
        G=None
        return data



    def B2A(B,w=None,normalize_type="full"):
        #for alpha forward , we only have the diagonal value for adj and deg and laplacian,and they are thesame
        if w is None:
            if normalize_type=="edge":
                DE=diag((B ).sum(0)** (-1))
                L_alpha=None
                A_beta=B @ (DE) @ B.T
                I = identity(A_beta.shape, device=B.device)
                A_beta+=I
                #
                D_beta=diag(A_beta.sum(1))
            ####
            
            elif normalize_type=="full":

                DE=diag(torch.pow((B ).sum(0),-1))
                L_alpha=None
                A_beta=B @ (DE) @ B.T
                I = identity(A_beta.shape, device=B.device)
                A_beta+=I
                #
                D_beta=diag(A_beta.sum(1))
            ####
                ##renormalization
                A_beta=D_beta**(-1/2) @ A_beta @ D_beta**(-1/2) 
                D_beta=diag(A_beta.sum(1))
            elif normalize_type=="none":
                L_alpha=None
                A_beta=B  @ B.T
                I = identity(A_beta.shape, device=B.device)
                #
                A_beta+=I
                D_beta=diag(A_beta.sum(1))
            elif normalize_type=="node":
                L_alpha=None
                A_beta=B  @ B.T
                I = identity(A_beta.shape, device=B.device)
                #
                A_beta+=I
                D_beta=diag(A_beta.sum(1))
                A_beta=D_beta**(-1/2) @ A_beta @ D_beta**(-1/2) 
                D_beta=diag(A_beta.sum(1))
        else:
            if normalize_type=="edge":
                DE=diag((B ).sum(0)** (-1))
                DE=DE@w
                L_alpha=None
                A_beta=B @ (DE ) @ B.T
                I = identity(A_beta.shape, device=B.device)
                A_beta+=I
                #
                D_beta=diag(A_beta.sum(1))
            ####
            
            elif normalize_type=="full":
                
                DE=diag(torch.pow((B ).sum(0),-1))
                DE=DE@w
                L_alpha=None
                A_beta=B @ (DE) @ B.T
                I = identity(A_beta.shape, device=B.device)
                A_beta+=I
                #
                D_beta=diag(A_beta.sum(1))
            ####
                ##renormalization
                A_beta=D_beta**(-1/2) @ A_beta @ D_beta**(-1/2) 
                D_beta=diag(A_beta.sum(1))
            elif normalize_type=="none":

                L_alpha=None
                A_beta=B @ w @ B.T
                I = identity(A_beta.shape, device=B.device)
                #
                A_beta+=I
                D_beta=diag(A_beta.sum(1))
            elif normalize_type=="node":
                L_alpha=None
                A_beta=B @w @ B.T
                I = identity(A_beta.shape, device=B.device)
                #
                A_beta+=I
                D_beta=diag(A_beta.sum(1))
                A_beta=D_beta**(-1/2) @ A_beta @ D_beta**(-1/2) 
                D_beta=diag(A_beta.sum(1))
        return L_alpha,A_beta,D_beta,I
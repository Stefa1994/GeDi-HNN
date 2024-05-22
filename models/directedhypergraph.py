import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.nn.conv import MessagePassing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP


class complex_relu_layer(nn.Module):
    """The complex ReLU layer from the `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.
    """
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real:torch.FloatTensor, img:torch.FloatTensor):
        """
        Complex ReLU function.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real:torch.FloatTensor, img:torch.FloatTensor):
        """
        Making a forward pass of the complex ReLU layer.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        real, img = self.complex_relu(real, img)
        return real, img

class complex_relu_layer_different(nn.Module):
    """
    The complex ReLU layer for quaternion where a function is applied specifically to each components
    """
    def __init__(self, ):
        super(complex_relu_layer_different, self).__init__()
    
    def complex_relu(self, real:torch.FloatTensor, imag_i:torch.FloatTensor):
        """
        Complex ReLU function.
        
        Arg types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        mask_r = 1.0*(real >= 0)
        mask_i = 1.0*(imag_i>= 0)
        return mask_r*real, mask_i*imag_i

    def forward(self, real:torch.FloatTensor, imag_i:torch.FloatTensor):
        """
        Making a forward pass of the complex ReLU layer.
        
        Arg types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag_1, imag_2, imag_3 (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        real, imag_i = self.complex_relu(real, imag_i)
        return real, imag_i

class SigMaNetConv(MessagePassing):
    

    def __init__(self, in_channels:int, out_channels:int, K:int, i_complex:bool=False, gcn:bool=True,
                 normalization:str='sym', bias:bool=True, edge_index=None, norm_real=None, norm_imag=None,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SigMaNetConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym'], 'Invalid normalization'
        kwargs.setdefault('flow', 'target_to_source')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.gcn = gcn
        #if gcn: # devo eliminare i pesi creati per moltiplicarli con il self-loop e creo solo un peso nel caso Theta moltiplica tutto [(I + A)\Theta]
        #    self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
        #else:
        self.weight = Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.i_complex = i_complex

        #Inserisco qui i valori di edge index, norm_real e norm_imagla creazione i valori come self
        self.edge_index = edge_index
        self.norm_real = norm_real
        self.norm_imag = norm_imag

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    
  # Possiamo utilizzare  questa funzione per elaborare la parte Tx0=I
    def process(self, mul_L_real, mul_L_imag, weight, X_real, X_imag):
        #data = torch.spmm(mul_L_real, X_real) sparse matrix
        Tx_0_real_real = torch.spmm(mul_L_real, X_real)
        real_real = torch.matmul(Tx_0_real_real, weight) 
        Tx_0_imag_imag = torch.matmul(mul_L_imag, X_imag)
        imag_imag = torch.matmul(Tx_0_imag_imag, weight) 

        Tx_0_real_imag = torch.matmul(mul_L_imag, X_real) # L_imag e x_reale --> real_imag
        real_imag = torch.matmul(Tx_0_real_imag, weight)
        Tx_0_imag_real = torch.matmul(mul_L_real, X_imag) # L_real e x_imag --> imag_real
        imag_real = torch.matmul(Tx_0_imag_real, weight)
        return real_real,Tx_0_real_real, imag_imag, Tx_0_imag_imag, imag_real, Tx_0_imag_real, real_imag, Tx_0_real_imag #torch.stack([real, imag])

    def forward(
        self,
        x_real: torch.FloatTensor, 
        x_imag: torch.FloatTensor, 
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the SigMaNet Convolution layer.
        
        Arg types:
            * x_real, x_imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long TensSor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * out_real, out_imag (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (N_nodes, F_out).
        """
        
        self.n_dim = x_real.shape[0]        
        # no correction of the sign since the output is already positive
        norm_imag = self.norm_imag 
        norm_real =  self.norm_real
        edge_index = self.edge_index


        #if self.gcn:
        #        # Nuovo codice con i cambi opportuni
        #Tx_1_real_real = self.propagate(edge_index, x=x_real, norm=norm_real, size=None).to(torch.float) # x_real - norm_real
            #print("Tx_1_real_real", Tx_1_real_real)
        #out_real_real = torch.matmul(Tx_1_real_real, self.weight[0])
        #        #print("output_real_real", out_real_real)
        #Tx_1_imag_imag = self.propagate(edge_index, x=x_imag, norm=norm_imag, size=None).to(torch.float) # x_imag - norm_imag
        #        #print("Tx_1_imag_imag", Tx_1_imag_imag)
        #out_imag_imag = torch.matmul(Tx_1_imag_imag, self.weight[0])
        #        #print("output_imag_imag", out_imag_imag)
        #Tx_1_imag_real = self.propagate(edge_index, x=x_imag, norm=norm_real, size=None).to(torch.float) # x_imag - norm_real
        #out_imag_real = torch.matmul(Tx_1_imag_real, self.weight[0])
        #Tx_1_real_imag = self.propagate(edge_index, x=x_real, norm=norm_imag, size=None).to(torch.float) # x_real - norm_imag
        #out_real_imag = torch.matmul(Tx_1_real_imag, self.weight[0])
        
        #out_real = out_real_real - out_imag_imag
        #out_imag = out_imag_real + out_real_imag   
        
        #else:
        
        
        if self.i_complex:
            i_real = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.ones(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
            i_imag = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.ones(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
        else:
            i_real = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.ones(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
            i_imag = torch.sparse_coo_tensor( (np.arange(self.n_dim), np.arange(self.n_dim)), np.zeros(self.n_dim), [self.n_dim, self.n_dim], dtype=torch.float32).to( device=x_real.device)
        out_real_real, Tx_0_real_real, out_imag_imag, Tx_0_imag_imag, \
                out_imag_real, Tx_0_imag_real, out_real_imag, Tx_0_real_imag = self.process(i_real, i_imag, self.weight[0], x_real, x_imag)
            # Nuovo codice con i cambi opportuni
        Tx_1_real_real = self.propagate(edge_index, x=x_real, norm=norm_real, size=None).to(torch.float) # x_real - norm_real
                    #print("Tx_1_real_real", Tx_1_real_real)
        out_real_real = out_real_real + torch.matmul(Tx_1_real_real, self.weight[1])
                   #print("output_real_real", out_real_real)
        Tx_1_imag_imag = self.propagate(edge_index, x=x_imag, norm=norm_imag, size=None).to(torch.float) # x_imag - norm_imag
                    #print("Tx_1_imag_imag", Tx_1_imag_imag)
        out_imag_imag = out_imag_imag + torch.matmul(Tx_1_imag_imag, self.weight[1])
                    #print("output_imag_imag", out_imag_imag)
        Tx_1_imag_real = self.propagate(edge_index, x=x_imag, norm=norm_real, size=None).to(torch.float) # x_imag - norm_real
        out_imag_real = out_imag_real + torch.matmul(Tx_1_imag_real, self.weight[1])
        Tx_1_real_imag = self.propagate(edge_index, x=x_real, norm=norm_imag, size=None).to(torch.float) # x_real - norm_imag
        out_real_imag = out_real_imag + torch.matmul(Tx_1_real_imag, self.weight[1])
        #
        out_real = out_real_real - out_imag_imag
        out_imag = out_imag_real + out_real_imag        


        if self.bias is not None:
            out_real += self.bias
            out_imag += self.bias

        return out_real, out_imag


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)


class Future_node_classification(nn.Module):
    def __init__(self, num_features:int, hidden:int=2, K:int=1, label_dim:int=2, \
        activation:bool=True, layer:int=2, dropout:float=0.5, normalization:str='sym', gcn:bool=True,\
        i_complex:bool=True, other_complex:bool=False, edge_index=None, norm_real=None, norm_imag=None, args=None):
        super(Future_node_classification, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(SigMaNetConv(in_channels=num_features, out_channels=hidden, K=K,\
                                  i_complex=i_complex,  normalization=normalization, edge_index=edge_index,\
            norm_real=norm_real, norm_imag=norm_imag, gcn=gcn))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer() #complex_relu_layer() #complex_relu_layer_different()# complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(SigMaNetConv(in_channels=hidden, out_channels=hidden, K=K, \
            i_complex=i_complex, normalization=normalization, \
            edge_index=edge_index, norm_real=norm_real, norm_imag=norm_imag))

        
        self.Chebs = chebs
        last_dim = 2
        self.classifier = MLP(in_channels=args.MLP_hidden*last_dim,
            hidden_channels=args.Classifier_hidden,
            out_channels=label_dim,
            num_layers=args.Classifier_num_layers,
            dropout=args.dropout,
            Normalization=args.normalization,
            InputNorm=False)

        #self.Conv = nn.Conv1d(hidden*last_dim, label_dim, kernel_size=1)
        #self.Conv2 = nn.Conv1d(hidden, label_dim, kernel_size=1)
        self.dropout = dropout
        self.other_complex = other_complex
    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        #self.Conv.reset_parameters()
        #self.Conv2.reset_parameters()
        self.classifier.reset_parameters()


    def forward(self, data):
        """
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * data (graph as input) - Edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        if self.other_complex:
            real, imag = data.x, data.x
        else:
            real = data.x
            imag = torch.zeros(data.x.size(), device=data.x.device)

        # No skip connection    
        #for cheb in self.Chebs:
        #    real, imag = cheb(real, imag)
        #    if self.activation:
        #        real, imag = self.complex_relu(real, imag)
        
        for ii, cheb in enumerate(self.Chebs):
            real_prev, imag_prev = real, imag  # Store previous values
            real, imag = cheb(real, imag)
            if self.activation:
                real, imag = self.complex_relu(real, imag)
            # Add skip connection
            if ii != 0:
                real += real_prev
                imag += imag_prev

        x = torch.cat((real, imag), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.classifier(x)
        #x = x.unsqueeze(0)
        #x = x.permute((0,2,1))
        #x = self.Conv(x)
        #x = self.Conv2(x)
        #x = x.permute((0,2,1)).squeeze()
        x = F.log_softmax(x, dim=1)
        return x




from typing import Optional
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import scipy
from torch_geometric.typing import OptTensor


def get_specific(vector, device):
    vector = vector.tocoo()
    row = torch.from_numpy(vector.row).to(torch.long)
    col = torch.from_numpy(vector.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0).to(device)
    edge_weight = torch.from_numpy(vector.data).to(device)
    return edge_index, edge_weight



def get_specific(vector, device):
    #vector = #vector.tocoo()
    #row = vector.indices()[0].to(torch.long) #vector.row.to(torch.long)
    #col = vector.indices()[1].to(torch.long) #vector.col.to(torch.long)
    edge_index = vector.indices().to(device) #torch.stack([row, col], dim=0).to(device)
    edge_weight = vector.values().to(device)
    return edge_index, edge_weight

from typing import Optional
from scipy.sparse import coo_matrix

def get_Laplacian_complited( edge_index = torch.LongTensor, edge_weight : Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None,
                  return_lambda_max: bool = False):
    
    '''
    This Laplacian is designed to work with the strange equation I designed
    Dv^-1/2((Ht * D_e^-1 *H_s.T) + (Hs * D_e^-1 * H_t^*) + I) Dv^-1/2
    We are not using it
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    row, col = edge_index
    size_col = max(col)
    if edge_weight is None:
        H = torch.sparse.FloatTensor(indices=edge_index,values=torch.ones(len(edge_index.T), device=device),size=(num_nodes, size_col),).cpu().coalesce()
    else:
        H = torch.sparse.FloatTensor(
    indices=edge_index,
    values=edge_weight,
    size=(num_nodes, size_col),
    ).cpu().coalesce()
    
    # estrazione della vertex degree (sommo su ogni riga) e dell'edge degree
    # Sum the real and imaginary parts element-wise
    # Node degree
    if edge_weight is None:
        node_degree = torch.abs(H.to_dense().sum(axis=1).real)   
    else:
        node_degree = torch.abs(H.to_dense().sum(axis=1).real) + torch.abs(H.to_dense().sum(axis=1).imag) # + identity  
    node_degree[node_degree == 0]= 1
    node_degree_inv_sqrt = torch.pow(node_degree, -0.5)
    node_degree_inv_sqrt[node_degree_inv_sqrt == float('inf')]= 0
    Dv = coo_matrix((node_degree_inv_sqrt, (np.arange(H.size()[0]), np.arange(H.size()[0]))), shape=(H.size()[0], H.size()[0]), dtype=np.float32).todense()
    print('Dimension Dv:', Dv.shape)
    if edge_weight is None:
        Dv = torch.tensor(Dv, dtype=torch.float)
    else:
        Dv = torch.tensor(Dv, dtype=torch.complex64)

    # Edge degree
    if edge_weight is None:
        edge_degree = torch.abs(H.to_dense().sum(axis=0).real) 
    else:
        edge_degree = torch.abs(H.to_dense().sum(axis=0).real) + torch.abs(H.to_dense().sum(axis=0).imag) # + identity

    edge_degree[edge_degree == 0]= 1
    edge_degree_inv_sqrt = torch.pow(edge_degree, -1)
    edge_degree_inv_sqrt[edge_degree_inv_sqrt == float('inf')]= 0
    De = coo_matrix((edge_degree_inv_sqrt, (np.arange(H.size()[1]), np.arange(H.size()[1]))), shape=(H.size()[1], H.size()[1]), dtype=np.float32).todense()
    if edge_weight is None:
        De = torch.tensor(De, dtype=torch.float)
    else:
        De = torch.tensor(De, dtype=torch.complex64)

    print('Dimension De:', De.shape)


    identity = coo_matrix( (np.ones(H.size()[0]), (np.arange(H.size()[0]), np.arange(H.size()[0]))), shape=(H.size()[0], H.size()[0]), dtype=np.float32).todense()
    # Costruzione di al normalizzata Dv^-1/2((Ht * D_e^-1 *H_s.T) + (Hs * D_e^-1 * H_t^*) + I) Dv^-1/2
    L = torch.mm(torch.mm( Dv, torch.mm(torch.mm(H.to_dense(), De.to_dense()), torch.conj(H.T.to_dense()))  ), Dv.to_dense()) + identity

    # Aggiungere identity matrix
    #L_norm = torch.to_numpy(L).to_sparse()

    edge_index, edge_weight = get_specific(L.to_sparse(), device)
    if torch.is_complex(edge_weight):
        return edge_index, edge_weight.real, edge_weight.imag
    imag = torch.zeros(edge_weight.real.shape, device = device)
    return edge_index, edge_weight.real, imag
    #else:
    #    return edge_index, edge_weight.real, edge_weight.imag




def __norm__(
        edge_index,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        lambda_max,
        dtype: Optional[int] = None
    ):
        """
        Get  Sign-Magnetic Laplacian.
        
        Arg types:
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * num_nodes (int, Optional) - Node features.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * edge_index, edge_weight_real, edge_weight_imag (PyTorch Float Tensor) - Magnetic laplacian tensor: edge index, real weights and imaginary weights.
        """
        #edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight_real, edge_weight_imag = get_Laplacian_complited(
            edge_index, edge_weight, num_nodes=num_nodes) 
        lambda_max.to(edge_weight_real.device)

        edge_weight_real = (2.0 * edge_weight_real) / lambda_max
        edge_weight_real.masked_fill_(edge_weight_real == float("inf"), 0)

        #_, edge_weight_real = add_self_loops(
        #    edge_index, edge_weight_real, fill_value=-1.0, num_nodes=num_nodes
        #)
        assert edge_weight_real is not None

        edge_weight_imag = (2.0 * edge_weight_imag) / lambda_max
        edge_weight_imag.masked_fill_(edge_weight_imag == float("inf"), 0)

        #edge_index, edge_weight_imag = add_self_loops(
        #    edge_index, edge_weight_imag, fill_value=0, num_nodes=num_nodes )
        assert edge_weight_imag is not None
        return edge_index, edge_weight_real, edge_weight_imag



def process_magnetic_laplacian_standard(edge_index: torch.LongTensor, x_real: Optional[torch.Tensor] = None, edge_weight: Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  num_nodes: Optional[int] = None,
                  lambda_max=None,
                  return_lambda_max: bool = False,
):
    lambda_max = torch.tensor(2.0, dtype=x_real.dtype,
                                      device=x_real.device)
    assert lambda_max is not None
    edge_index, norm_real, norm_imag = __norm__(edge_index=edge_index,num_nodes=num_nodes,edge_weight= edge_weight, 
                                                lambda_max=lambda_max) 
    
    return edge_index, norm_real, norm_imag
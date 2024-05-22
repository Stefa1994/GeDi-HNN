from typing import Optional
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import scipy
from torch_geometric.typing import OptTensor
from typing import Optional
from scipy.sparse import coo_matrix

def get_specific(vector, device):
    vector = vector.tocoo()
    row = torch.from_numpy(vector.row).to(torch.long)
    col = torch.from_numpy(vector.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0).to(device)
    edge_weight = torch.from_numpy(vector.data).to(device)
    return edge_index, edge_weight



def get_Laplacian_complited( edge_index = torch.LongTensor, edge_weight : Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None,
                  return_lambda_max: bool = False):
    
    """
    This section encode the Laplacian
    I - Dv^-1/2( H D_e^-1 H^* + I) Dv^-1/2
    We use this equation    
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    if edge_weight is None or not torch.is_complex(edge_weight):
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)
    # incident matrix construction
    row, col = edge_index.cpu()
    size_col = max(col) + 1
    H = coo_matrix((edge_weight.cpu(), (row, col)), shape=(num_nodes, size_col), dtype=np.complex64)

    d_node = np.array(np.abs(H).sum(axis=1)) #.real + np.abs(H).sum(axis=1).imag) #[0] # node degree (sommo sulle righe)
    d_node[d_node == 0] = 1
    deg_node_inv_sqrt = np.power(d_node, -0.5)
    deg_node_inv_sqrt[deg_node_inv_sqrt == float('inf')]= 0
    Dv = coo_matrix((deg_node_inv_sqrt.flatten(), (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32)
        

    # vertex degree with identity matrix
    diag = coo_matrix( (np.ones(num_nodes), (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32).todense()
    degree_vert = coo_matrix((d_node.flatten(), (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32).todense()
    deg_node_inv_sqrt = np.power((degree_vert + diag).diagonal(), -0.5)
    deg_node_inv_sqrt[deg_node_inv_sqrt == float('inf')]= 0
    D_vn = coo_matrix((np.array(deg_node_inv_sqrt.flatten())[0], (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32)

    
    ##########################################
    ##### Edge degree  
    #############################################
    d_edge = np.array(np.abs(H).sum(axis=0).real + np.abs(H).sum(axis=0).imag)[0] # edge degree (sommo sulle colonne)
    d_edge[d_edge == 0] = 1 
    deg_edge_inv_sqrt = np.power(d_edge, -1) #deg_edge_inv_sqrt = 1/d_edge #np.power(d_edge, -1) or flatyten()
    deg_edge_inv_sqrt[deg_edge_inv_sqrt == float('inf')]= 0
    De = coo_matrix((deg_edge_inv_sqrt.flatten(), (np.arange(size_col), np.arange(size_col))), shape=(size_col, size_col), dtype=np.float32)



    
    A = H.dot(De).dot(np.conjugate(H).T)

    L =  Dv.dot(A).dot(Dv) 

    diag_n = coo_matrix((L.diagonal(), (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32)



    A_diag = coo_matrix((A.diagonal(), (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32)

    A_out =  A - A_diag
    A_n = A_out + diag_n 
    L2 =  D_vn.dot(A_n).dot(D_vn)

    edge_index, edge_weight = get_specific(L2, device)
    return edge_index, edge_weight.real, edge_weight.imag




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
        edge_index, edge_weight_real, edge_weight_imag = get_Laplacian_complited(
            edge_index, edge_weight, num_nodes=num_nodes) 
        lambda_max.to(edge_weight_real.device)

        edge_weight_real = (2.0 * edge_weight_real) / lambda_max
        edge_weight_real.masked_fill_(edge_weight_real == float("inf"), 0)

        assert edge_weight_real is not None

        edge_weight_imag = (2.0 * edge_weight_imag) / lambda_max
        edge_weight_imag.masked_fill_(edge_weight_imag == float("inf"), 0)

        assert edge_weight_imag is not None
        return edge_index, edge_weight_real, edge_weight_imag



def process_magnetic_laplacian_sparse(edge_index: torch.LongTensor, x_real: Optional[torch.Tensor] = None, edge_weight: Optional[torch.Tensor] = None,
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
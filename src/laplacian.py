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

def get_Laplacian( edge_index = torch.LongTensor, edge_weight : Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None,
                  return_lambda_max: bool = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    H = torch.sparse.FloatTensor(
    indices=edge_index,
    values=edge_weight,
    size=(num_nodes, num_nodes),
    ).cpu().coalesce()
    #edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    # We receive as input an incident matrix [N x M] with edge_index and weight


    # da sistemare
    #if edge_weight is None:
    #    edge_weight = torch.eye(H.size()[1], dtype=dtype,
    #                             device=edge_index.device)

    #num_nodes = #maybe_num_nodes(edge_index, num_nodes)
    #row, col = edge_index.cpu()

    #A = coo_matrix((edge_weight.cpu(), (row, col)), shape=(size, size), dtype=np.float32)
    
    # estrazione delle due matrici, target e sorgente    

    # Create masks for values equal to 1 and -1j
    mask_1 = (H.values().real == 1) & (H.values().imag == 0)
    mask_minus_1j = (H.values().real == 0) & (H.values().imag == -1)
    
    # Extract matrices with values equal to 1 and -1j
    H_s = torch.sparse.FloatTensor(
        indices=H.indices()[:, mask_1],
        values=torch.ones_like(H.values()[mask_1]),
        size=H.size(),
    )
    
    H_t = torch.sparse.FloatTensor(
        indices=H.indices()[:, mask_minus_1j],
        values=torch.ones_like(H.values()[mask_minus_1j]) * (-1j),
        size=H.size(),
    )


    # estrazione della vertex degree (sommo su ogni riga) e dell'edge degree
    # Sum the real and imaginary parts element-wise
    # Node degree
    node_degree = torch.abs(H.to_dense().sum(axis=1).real) + torch.abs(H.to_dense().sum(axis=1).imag) # + identity  
    node_degree[node_degree == 0]= 1
    node_degree_inv_sqrt = torch.pow(node_degree, -0.5)
    node_degree_inv_sqrt[node_degree_inv_sqrt == float('inf')]= 0
    Dv = coo_matrix((node_degree_inv_sqrt, (np.arange(H.size()[0]), np.arange(H.size()[0]))), shape=(H.size()[0], H.size()[0]), dtype=np.float32).todense()
    Dv = torch.tensor(Dv, dtype=torch.complex128)
    # Edge degree
    edge_degree = torch.abs(H.to_dense().sum(axis=0).real) + torch.abs(H.to_dense().sum(axis=0).imag) # + identity
    edge_degree[edge_degree == 0]= 1
    edge_degree_inv_sqrt = torch.pow(edge_degree, -1)
    edge_degree_inv_sqrt[edge_degree_inv_sqrt == float('inf')]= 0
    De = coo_matrix((edge_degree_inv_sqrt, (np.arange(H.size()[1]), np.arange(H.size()[1]))), shape=(H.size()[1], H.size()[1]), dtype=np.float32).todense()
    De = torch.tensor(De, dtype=torch.complex128)
    
    identity = coo_matrix( (np.ones(H.size()[0]), (np.arange(H.size()[0]), np.arange(H.size()[0]))), shape=(H.size()[0], H.size()[0]), dtype=np.float32).todense()
    # Costruzione di al normalizzata Dv^-1/2((Ht * D_e^-1 *H_s.T) + (Hs * D_e^-1 * H_t^*) + I) Dv^-1/2

    L = torch.mm(torch.mm( Dv, torch.mm(torch.mm(H_t.to_dense(), De.to_dense()), H_s.T.to_dense()) +   torch.mm(torch.mm(H_s.to_dense(), De.to_dense()), torch.conj(H_t.T.to_dense()))  ), Dv.to_dense()) + identity

    # Aggiungere identity matrix
    #L_norm = torch.to_numpy(L).to_sparse()

    edge_index, edge_weight = get_specific(L.to_sparse(), device)
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
        #edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight_real, edge_weight_imag = get_Laplacian(
            edge_index, edge_weight, num_nodes=num_nodes) #, gcn, net_flow, edge_weight, normalization, dtype, num_nodes  )
        lambda_max.to(edge_weight_real.device)

        edge_weight_real = (2.0 * edge_weight_real) / lambda_max
        edge_weight_real.masked_fill_(edge_weight_real == float("inf"), 0)

        _, edge_weight_real = add_self_loops(
            edge_index, edge_weight_real, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight_real is not None

        edge_weight_imag = (2.0 * edge_weight_imag) / lambda_max
        edge_weight_imag.masked_fill_(edge_weight_imag == float("inf"), 0)

        edge_index, edge_weight_imag = add_self_loops(
            edge_index, edge_weight_imag, fill_value=0, num_nodes=num_nodes )
        assert edge_weight_imag is not None
        return edge_index, edge_weight_real, edge_weight_imag



def process_magnetic_laplacian(edge_index: torch.LongTensor, x_real: Optional[torch.Tensor] = None, edge_weight: Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  num_nodes: Optional[int] = None,
                  lambda_max=None,
                  return_lambda_max: bool = False,
):
    #if normalization != 'sym' and lambda_max is None:        
    #    _, _, _, lambda_max =  get_Laplacian(
    #    edge_index, gcn, edge_weight, None, return_lambda_max=True )

    #if lambda_max is None:
    #    lambda_max = torch.tensor(2.0, dtype=x_real.dtype, device=x_real.device)
    #if not isinstance(lambda_max, torch.Tensor):
    lambda_max = torch.tensor(2.0, dtype=x_real.dtype,
                                      device=x_real.device)
    assert lambda_max is not None
    node_dim = -2
    edge_index, norm_real, norm_imag = __norm__(edge_index=edge_index,num_nodes=num_nodes,edge_weight= edge_weight, 
                                                lambda_max=lambda_max) # type: ignore
                                         #lambda_max, dtype=x_real.dtype)
    
    return edge_index, norm_real, norm_imag
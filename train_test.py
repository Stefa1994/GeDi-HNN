import os, sys
import math, time, random
import pickle
import argparse, configargparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
#from transformers.optimization import  Adafactor

from tqdm import tqdm

from models import SetGNN, HCHA, HNHN, HyperGCN, HyperSAGE, \
    LEGCN, UniGCNII, HyperND, EquivSetGNN, Future_node_classification, GCNModel, Future_node_classification_new
from scipy.sparse import coo_matrix
#from data import read_data, read_data_pub

from src import process_magnetic_laplacian_sparse
import datasets_direction
import datasets

import utils

'''
For running the detaset expect for Telegram, Cornell, Texas, Wisconsin
'''

def vertex_degree(edge_index, num_nodes):
    # incident matrix construction
    row, col = edge_index
    size_col = max(col) + 1
    H = coo_matrix((np.ones(len(edge_index.T)), (row, col)), shape=(num_nodes, size_col), dtype=np.float32).tocsr()
    degree = torch.from_numpy(np.sum(np.abs(H), axis = 1)).float()
    return degree

@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, loss_fn=None, return_out=False):
    model.eval()
    out = model(data)
    out = F.log_softmax(out, dim=1)

    train_acc = evaluator.eval(data.y[split_idx['train']], out[split_idx['train']])['acc']
    valid_acc = evaluator.eval(data.y[split_idx['valid']], out[split_idx['valid']])['acc']
    test_acc = evaluator.eval(data.y[split_idx['test']], out[split_idx['test']])['acc']

    ret_list = [train_acc, valid_acc, test_acc]

    # Also keep track of losses
    if loss_fn is not None:
        train_loss = loss_fn(out[split_idx['train']], data.y[split_idx['train']])
        valid_loss = loss_fn(out[split_idx['valid']], data.y[split_idx['valid']])
        test_loss = loss_fn(out[split_idx['test']], data.y[split_idx['test']])
        ret_list += [train_loss, valid_loss, test_loss]

    if return_out:
        ret_list.append(out)

    return ret_list

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')

    if args.method not in ['HyperGCN', 'HyperSAGE']:
        transform = torch_geometric.transforms.Compose([datasets.AddHypergraphSelfLoops()])
    else:
        transform = None
    
    if not args.directed:
        print('undirected')
        data = datasets.HypergraphDataset(root=args.data_dir, name=args.dname, path_to_download=args.raw_data_dir,
        feature_noise=args.feature_noise, transform=transform, second_name=args.second_name).data
    else:
        print('directed')
        data = datasets_direction.HypergraphDataset(root=args.data_dir, name=args.dname, path_to_download=args.raw_data_dir,
        feature_noise=args.feature_noise, transform=transform, second_name=args.second_name).data 
    if data.x is None:
        data.x = vertex_degree(data.edge_index, data.num_nodes)
    
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = SetGNN.norm_contruction(data, option=args.normtype)
    elif args.method == 'HNHN':
        data = HNHN.generate_norm(data, args)
    elif args.method == 'HyperSAGE':
        data = HyperSAGE.generate_hyperedge_dict(data)
    elif args.method == 'LEGCN':
        data = LEGCN.line_expansion(data)
    elif args.method =='PhenomNN':
        data = GCNModel.data_creation(data, args)
    elif args.method =='GEDI':
        edge_index, norm_real, norm_imag = process_magnetic_laplacian_sparse(edge_index=data.edge_index, x_real=data.x, edge_weight=data.edge_weight, \
        normalization = 'sym', num_nodes=data.num_nodes,return_lambda_max = False)
    data = data.to(device)

    # Get splits
    split_idx_lst = []
    for run in range(args.runs):
        if args.dname in ['WikiCS']:
            split_idx = utils.rand_train_test_idx_2(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop, balance=True) 
        else:
            split_idx = utils.rand_train_test_idx_2(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop, balance=False) 
        split_idx_lst.append(split_idx)


   # define the model

    if args.method == 'AllSetTransformer':
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method == 'AllDeepSets':
        args.AllSet_PMA = False
        args.aggregate = 'add'
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method in ['HGNN', 'HCHA']:
        args.method = 'HCHA_att_sym' if args.HCHA_att and args.HCHA_symdegnorm else \
              'HCHA_att' if args.HCHA_att else args.method
        model = HCHA(data.num_features, data.num_classes, args)
    elif args.method in 'HNHN':
        model = HNHN(data.num_features, data.num_classes, args)
    elif args.method in 'HyperGCN':
        model = HyperGCN(data.num_features, data.num_classes, args)
    elif args.method == 'HyperSAGE':
        model = HyperSAGE(data.num_features, data.num_classes, args)
    elif args.method == 'LEGCN':
        model = LEGCN(data.num_features, data.num_classes, args)
    elif args.method == 'UniGCNII':
        model = UniGCNII(data.num_features, data.num_classes, args)
    elif args.method == 'HyperND':
        model = HyperND(data.num_features, data.num_classes, args)
    elif args.method == 'EDGNN':
        model = EquivSetGNN(data.num_features, data.num_classes, args)
    elif args.method == 'PhenomNN':
        _ = GCNModel(nfeat=data.num_features, nhid=args.MLP_hidden, nclass=data.num_classes, nhidlayer=args.nhidden,
                    dropout=args.dropout, baseblock='phenomnn', nbaselayer=args.nbaseblocklayer,
                    args=args)
    elif args.method == 'GeDi':
        model = Future_node_classification(K=1, num_features=data.num_features, hidden=args.MLP_hidden, label_dim=data.num_classes,     # hidden=256 dropout= 0.3
                            i_complex = False,  layer=args.nconv, other_complex=args.other_complex, edge_index=edge_index,\
                           norm_real=norm_real, norm_imag=norm_imag, dropout=args.dropout, gcn=False, args=args)
    else:
        raise ValueError(f'Undefined model name: {args.method}')
    
    
    if not args.method == 'PhenomNN':
        model = model.to(device)
        num_params = count_parameters(model)
        print("# Params:", num_params)
    
    print("# Classes:", data.num_classes)
    logger = utils.Logger(args.runs, args)
    loss_fn = nn.NLLLoss()
    evaluator = utils.NodeClsEvaluator()
    if (not args.directed) and (args.method == 'GeDi'):
        args.method = 'GeDi_undirected' 
    runtime_list = []
    for run in range(args.runs):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        if args.method =="PhenomNN":
            model = GCNModel(nfeat=data.num_features, nhid=args.MLP_hidden, nclass=data.num_classes, nhidlayer=args.nhidden,
                    dropout=args.dropout, baseblock='phenomnn', nbaselayer=args.nbaseblocklayer,
                    args=args).to(device)
            _,A_beta,D_beta,I=GCNModel.B2A(data.B,normalize_type="node")
            L_alpha,A_gamma,D_gamma,_= GCNModel.B2A(data.B,normalize_type="full")
            data.H=[A_beta,A_gamma]
            data.G=[D_beta,D_gamma,I]
            num_params = count_parameters(model)
        model.reset_parameters()
        best_val = float('-inf')

        #optimizer = Adafactor(model.parameters(), weight_decay=0)#, lr=0.01, relative_step= False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in range(args.epochs):
            # Training loop
            model.train()
            optimizer.zero_grad()
            out = model(data)    #(data)
            out = F.log_softmax(out, dim=1)
            loss = loss_fn(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()
            # Evaluation and logging
            result = evaluate(model, data, split_idx, evaluator, loss_fn)
            logger.add_result(run, *result[:3])
            if epoch % args.display_step == 0 and args.display_step > 0 and 100 * result[1] > best_val:
                print(f'Run: {run:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test Acc: {100 * result[2]:.2f}%')
                best_val = 100 * result[1]
        end_time = time.time()
        runtime_list.append(end_time - start_time)

    logger.print_statistics(args=args)

    ## Save results ###
    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)
#
    best_val, best_test = logger.print_statistics()
    res_root = 'hyperparameter_tunning'
    if not os.path.isdir(res_root):
        os.makedirs(res_root)
#
    filename = f'{res_root}/{args.dname}.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'{args.method}'
        cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
        cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
        cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s' 
        cur_line += f',{avg_time//60}min{(avg_time % 60):.2f}s'
        cur_line += f'\n'
        write_obj.write(cur_line)
#
    all_args_file = f'{res_root}/all_args_{args.dname}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')
#    
    print('All done! Exit python code')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', is_config_file=True)

    # Dataset specific arguments
    parser.add_argument('--dname', default='walmart-trips-100')
    parser.add_argument('--data_dir', type=str, required=False)
    parser.add_argument('--raw_data_dir', type=str, required=False)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--feature_noise', default='1', type=str, help='std for synthetic feature noise')
    parser.add_argument('--normtype', default='all_one', choices=['all_one','deg_half_sym'])
    parser.add_argument('--add_self_loop', action='store_false')
    parser.add_argument('--exclude_self', action='store_true', help='whether the he contain self node or not')
    parser.add_argument('--second_name', type=str, default='')
    parser.add_argument('--directed', type=bool, default=False)

    # Training specific hyperparameters
    parser.add_argument('--epochs', default=500, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=10, type=int)
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--input_dropout', default=0.2, type=float)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--display_step', type=int, default=50)

    # For saving porpuse
    parser.add_argument('--config_number', type=int, default=0)
    parser.add_argument("--save_result", action="store_true", default=False)

    # Model common hyperparameters
    parser.add_argument('--method', default='EDGNN', help='model type')
    parser.add_argument('--All_num_layers', default=2, type=int, help='number of basic blocks')
    parser.add_argument('--MLP_num_layers', default=2, type=int, help='layer number of mlps')
    parser.add_argument('--MLP_hidden', default=64, type=int, help='hidden dimension of mlps')
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normalization', default='ln', choices=['bn','ln','None'])
    parser.add_argument('--activation', default='relu', choices=['Id','relu', 'prelu'])
    
    # Args for GeDi
    parser.add_argument('--other_complex', action='store_true', default=False)
    parser.add_argument('--nconv', default=2, type=int)

    
    # Args for EDGNN
    parser.add_argument('--MLP2_num_layers', default=-1, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=-1, type=int, help='layer number of mlp3')
    parser.add_argument('--edconv_type', default='EquivSet', type=str, choices=['EquivSet', 'JumpLink', 'MeanDeg', 'Attn', 'TwoSets'])
    parser.add_argument('--restart_alpha', default=0.5, type=float)

    # Args for AllSet
    parser.add_argument('--AllSet_input_norm', default=True)
    parser.add_argument('--AllSet_GPR', action='store_false')  # skip all but last dec
    parser.add_argument('--AllSet_LearnMask', action='store_false')
    parser.add_argument('--AllSet_PMA', action='store_true')
    parser.add_argument('--AllSet_num_heads', default=1, type=int)
    # Args for CEGAT
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    # Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    # Args for HyperSAGE
    parser.add_argument('--HyperSAGE_power', default=1., type=float)
    parser.add_argument('--HyperSAGE_num_sample', default=100, type=int)
    # Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    # Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    parser.add_argument('--HCHA_att', default=False, type=bool)
    # Args for UniGNN
    parser.add_argument('--UniGNN_use_norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    # Args for HyperND
    parser.add_argument('--HyperND_ord', default = 1., type=float)
    parser.add_argument('--HyperND_tol', default = 1e-4, type=float)
    parser.add_argument('--HyperND_steps', default = 100, type=int)

    # argument for phenomnn   
    parser.add_argument('--LP', action='store_true', default=False, help='Label propagation')
    parser.add_argument('--lam4', type=float, default=0, help='lam4.')
    parser.add_argument('--lam0', type=float, default=10, help='lam0.')
    parser.add_argument('--lam1', type=float, default=10, help='lam1.')
    parser.add_argument('--normalize_type', type=str, default="full", help='normalize type for phenomnn')
    parser.add_argument('--H', action='store_true', default=False, help='whether to use compatibility matrix in phenomnn')
    parser.add_argument('--HisI', action='store_true', default=False, help='if using H and H is I ')
    parser.add_argument('--notresidual', action='store_true', default=False, help='whether to use residual in H')
    parser.add_argument('--twoHgamma', action='store_true', default=False, help='whether to use two H for gamma matrix')
    parser.add_argument("--nbaseblocklayer", type=int, default=1,
                        help="The number of layers in each baseblock")  # same as '--layer' of gcnii
    parser.add_argument('--alp', type=float, default=1, help='alp')
    parser.add_argument('--prop_step', type=int, default=16, help='propagating steps')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')# non suo
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')# non suo
    parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.') 
    parser.add_argument('--sigma', type=float, default=-1, help='sigma for edge degree matirx.')
    parser.add_argument('--nhidden', default=2, type=int)


    args = parser.parse_args()

    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(AllSet_GPR=False)
    parser.set_defaults(AllSet_LearnMask=False)
    parser.set_defaults(AllSet_PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
    
    #     Use the line below for .py file
    args = parser.parse_args()
    #     Use the line below for notebook
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
    #main(**vars(args))
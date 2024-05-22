# GD_HNN

This repository contains the official PyTorch implementation of GD-HNN, including both its code and the code for running other Hypergraph Neural Networks.

## Enviroment Setup
The experiments were conducted under this specific environment:

1. Ubuntu 20.04.3 LTS
2. Python 3.8.10
3. CUDA 10.2
4. Torch 1.11.0 (with CUDA 10.2)


In addition, torch-scatter, torch-sparse and torch-geometric are needed to handle scattered graphs and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data. For these three packages, follow the official instructions for [torch-scatter](https://github.com/rusty1s/pytorch_scatter), [torch-sparse](https://github.com/rusty1s/pytorch_sparse), and [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

Pytorch Geometric Signed Directed [GitHub Pages](https://github.com/SherylHYX/pytorch_geometric_signed_directed) version 0.3.1 and Networkx version 2.8 must be installed.

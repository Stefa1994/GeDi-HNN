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

## Repository structure

The repository contains three folders:
- **data** contains the syntactic graphs without digos in *synthetic*, the synthetic graphs with digons in *synthetic_digons* and the WikiRfa dataset in *wikirfa*.
- **generative_graph** contains the code for creating the two different classes of synthetic graphs:
   -  *generative_synthetic_data* for graphs without digons
   -  *generate_synthetic_data_with_antiparallel* for graphs with digons 
- **src** contains all the model implementations used for running the experiments. Futhermore, it stores two other foldes **utils** and **layer**.

## Run code

```
cd src
python3 QuaterGCN.py --dataset dataset_nodes500_alpha0.05_beta0.2
python3 Edge_QuaterGCN.py --dataset dataset_nodes500_alpha0.05_beta0.2 --task three_class_digraph --noisy
python3 sing_link_prediction.py --dataset=bitcoin_alpha --task=four_class_signed_digraph --num_classes=4 --num_layers=2 --epochs=300 --dropout=0.5 --lr=1e-2
python3 sing_link_prediction.py --dataset=bitcoin_alpha --task=five_class_signed_digraph --num_classes=5 --num_layers=2 --epochs=300 --dropout=0.5 --lr=1e-2
```


## License

QuaNet is released under the [Apace 2.0 License](https://choosealicense.com/licenses/mit/)

# GeDi-HNN

This repository contains the official PyTorch implementation of GeDi-HNN, including both its code and the code for running other Hypergraph Neural Networks.

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
- **data** contains the original_data folder with the dataset divided in:
   1. **cocitation** folder for *citeseer*, *cora*, *pubmed*
   2. **mail** folder for *email-Enron*, *email-Eu*
   3. **synthetic** folder for the syntehtic dataset
   4. **other** folder to save the other datasets
- **model** contains the code for the model
- **src** contains the implementation of our Lapalcian

## Run code

Example for the training of GeDi-HNN

```
python3 train_test_1.py --method GeDi --dname telegram --second_name telegram --nconv 2 --Classifier_num_layers 2 --MLP_hidden 64 --Classifier_hidden 64 --wd 0.005 --epochs 500 --runs 10 --directed True --data_dir <data_path> --raw_data_dir <raw_data_path>


python3 train_test.py --method GeDi --dname Eu --second_name Eu --nconv 2 --Classifier_num_layers 2 --MLP_hidden 64 --Classifier_hidden 64 --wd 0.005 --epochs 500 --runs 10 --directed True --data_dir <data_path> --raw_data_dir <raw_data_path>
```

Note that ```--raw_data_dir``` is the full path to load raw data. The processed data will be saved into the path specified by ```--data_dir```.


## License

GeDi-HNN is released under the [Apace 2.0 License](https://choosealicense.com/licenses/mit/)

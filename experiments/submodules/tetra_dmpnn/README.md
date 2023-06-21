# Chirality-aware message passing networks
Custom aggregation functions for molecules with tetrahedral chirality ([arXiv](https://arxiv.org/abs/2012.00094))

## Requirements
* python (version>=3.7)
* pytorch (version>=1.14)
* rdkit (version>=2020.03.2)
* pytorch-geometric (version>=1.6.0)

## Installation
First, clone the repository:
`git clone https://github.com/PattanaikL/chiral_gnn`

Run `make conda_env` to create the conda environment. 
The script will request the user to enter one of the supported CUDA versions listed here: https://pytorch.org/get-started/locally/.
The script uses this CUDA version to install PyTorch and PyTorch Geometric. Alternatively, the user could manually follow the steps to install PyTorch Geometric here: https://github.com/rusty1s/pytorch_geometric/blob/master/.travis.yml.

## Usage
For the toy classification task, call the `train.py` script with the following parameters defined:

`python train.py --data_path data/d4_docking/d4_docking_rs.csv --split_path data/d4_docking/rs/split0.npy --task classification --log_dir ./test_run --gnn_type dmpnn --message tetra_permute_concat`


To train the model with the best-performing parameters, call the `train.py` script with the following parameters defined:

`python train.py --data_path data/d4_docking/d4_docking.csv --split_path data/d4_docking/full/split0.npy --log_dir ./test_run --gnn_type dmpnn --message tetra_permute_concat --global_chiral_features --chiral_features`

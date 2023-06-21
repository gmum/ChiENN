# ChiENN - experiments

This module contains code for running experiments with ChiENN and baselines models. It was adapted from [GraphGPS repository](https://github.com/rampasek/GraphGPS).


### Python environment setup with Conda

```bash
conda create -n chienn-experiments python=3.9
conda activate chienn-experiments

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb
pip install PyTDC
pip install chainer-chemistry
pip install schnetpack==1.0.1

conda clean --all
```


### Running ChiENN
```bash
conda activate chienn-experiments

# Running ChiENN with parameters tuned for binding_rank:
python main.py --cfg configs/models/ChiENN/binding_rank-ChiENN.yaml  wandb.use False
```


### Benchmarking ChiENN
To run a benchmark that tunes the hypeparameters on a validation set and then evaluates the model on a test set, use the `benchmark.py` script and configs from `configs/benchmarks/`:
```bash
conda activate chienn-experiments
# Run 3 repeats with seed=0,1,2:
python main.py --cfg configs/benchmarks/benchmark-binding_rank-ChiENN.json  --repeat 3  wandb.use False
```


### W&B logging
To use W&B logging, set `wandb.use True` and have a `chienn` entity set-up in your W&B account (or change it to whatever else you like by setting `wandb.entity` in `configs/models/common.yaml`).


## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{chienn,
  title={{ChiENN: Embracing Molecular Chirality with Graph Neural Networks}}, 
  author={Piotr Gaiński, Michał Koziarski, Jacek Tabor, Marek Śmieja},
  year={2023}
}
```
and the paper that introduced GraphGPS:
```bibtex
@article{rampasek2022GPS,
  title={{Recipe for a General, Powerful, Scalable Graph Transformer}}, 
  author={Ladislav Ramp\'{a}\v{s}ek and Mikhail Galkin and Vijay Prakash Dwivedi and Anh Tuan Luu and Guy Wolf and Dominique Beaini},
  journal={arXiv:2205.12454},
  year={2022}
}
```

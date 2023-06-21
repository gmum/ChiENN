import sys

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/GeometricTransformerMolecule')
from graphgps.dataset.utils import create_custom_loader
from main import custom_load_cfg

import graphgps  # noqa, register custom modules

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, set_cfg)
from torch_geometric import seed_everything

if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    cfg_file = args.cfg_file
    opts = args.opts

    # Load config
    set_cfg(cfg)
    custom_load_cfg(cfg=cfg, cfg_file=cfg_file, opts=opts)
    seed_everything(cfg.seed)
    create_custom_loader()

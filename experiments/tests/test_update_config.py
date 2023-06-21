import sys

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')
from torch_geometric.graphgym import cfg, set_cfg
from graphgps.utils import update_cfg


def test__update_cfg_works_correctly():
    set_cfg(cfg)
    cfg.share.dim_in = 0
    cfg.optim.base_lr = 0.0
    cfg.posenc_LapPE.eigen.max_freqs = 10
    setting = {
        'share.dim_in': 64,
        'optim.base_lr': 1.0,
        'posenc_LapPE.eigen.max_freqs': 3,
    }
    update_cfg(cfg, setting)

    assert cfg.share.dim_in == 64
    assert cfg.optim.base_lr == 1.0
    assert cfg.posenc_LapPE.eigen.max_freqs == 3

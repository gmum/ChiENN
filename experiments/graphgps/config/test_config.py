from yacs.config import CfgNode
from torch_geometric.graphgym.register import register_config


@register_config('test_cfg')
def dataset_cfg(cfg):
    """Test-specific config options.
    """

    cfg.test = CfgNode()

    # Sampling strategy for a test loader
    cfg.test.sampler = 'full_batch'

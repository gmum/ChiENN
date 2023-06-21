from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('custom_model_cfg')
def dataset_cfg(cfg):
    """Model-specific config options.
    """

    # Path to config used in ChIRo model.
    cfg.model.config_path = ""

    cfg.model.hidden_dim = 10

    # Method for coordinates selection in AtomDistance class
    cfg.model.coords_selection = "start"

    cfg.model.add_chienn_layer = False

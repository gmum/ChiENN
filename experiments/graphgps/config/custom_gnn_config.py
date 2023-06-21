from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False

    # Used in DMPNN class.
    cfg.gnn.tetra = CN()
    cfg.gnn.tetra.use = False
    cfg.gnn.tetra.message = "tetra_permute"
    cfg.gnn.layers = 5
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('chienn_cfg')
def chienn_cfg(cfg):
    """
    Config option for ChiENN model.
    """
    cfg.chienn = CN()

    # Parameters for message module of ChiENN
    cfg.chienn.message = CN()
    cfg.chienn.message.k_neighbors_embeddings_names = ['linear', 'linear', 'linear']
    cfg.chienn.message.final_embedding_name = 'ELU+linear'

    # Parameters for aggregation module of ChiENN
    cfg.chienn.aggregate = CN()
    cfg.chienn.aggregate.self_embedding_name = 'linear'
    cfg.chienn.aggregate.parallel_embedding_name = 'none'
    cfg.chienn.aggregate.aggregation = 'sum'
    cfg.chienn.aggregate.post_aggregation_embedding_name = 'ELU'

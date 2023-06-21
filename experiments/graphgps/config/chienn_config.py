from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('chienn_cfg')
def chienn_cfg(cfg):
    """
    Config option for ChiENN model.
    """
    cfg.chienn = CN()

    cfg.chienn.mask_non_chiral = False

    # Parameters for message module of ChiENN
    cfg.chienn.message = CN()
    cfg.chienn.message.single_direction = False
    cfg.chienn.message.shared_weights = False
    cfg.chienn.message.embedding_names = ['linear', 'linear']
    cfg.chienn.message.mask_by_in_degree = False
    cfg.chienn.message.aggregation = 'add'
    cfg.chienn.message.final_embedding_name = 'identity'
    cfg.chienn.message.sanity_check_in_degree = False

    # Parameters for aggregation module of ChiENN
    cfg.chienn.aggregate = CN()
    cfg.chienn.aggregate.double_aggregation = False
    cfg.chienn.aggregate.self_embedding_name = 'linear'
    cfg.chienn.aggregate.distinct_self = False
    cfg.chienn.aggregate.parallel_embedding_name = 'none'
    cfg.chienn.aggregate.aggregation = 'gat_attention'
    cfg.chienn.aggregate.embedding_name = 'identity'
    cfg.chienn.aggregate.final_embedding_name = 'ELU'
    cfg.chienn.aggregate.after_aggregation_embedding_name = 'identity'

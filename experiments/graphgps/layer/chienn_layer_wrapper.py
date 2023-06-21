import sys

from torch_geometric.data import Batch
from torch_geometric.graphgym import cfg

sys.path.append('..')
from chienn import ChiENNLayer


class ChiENNLayerWrapper(ChiENNLayer):
    """
    Wrapper for ChiENNLayer that loads the parameters form the graphgym config, making the initialization easier.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0, return_batch: bool = False):
        super().__init__(
            hidden_dim=hidden_dim,
            k_neighbors_embeddings_names=cfg.chienn.message.k_neighbors_embeddings_names,
            message_final_embedding_name=cfg.chienn.message.final_embedding_name,
            aggregation_name=cfg.chienn.aggregate.aggregation,
            self_embedding_name=cfg.chienn.aggregate.self_embedding_name,
            parallel_embedding_name=cfg.chienn.aggregate.parallel_embedding_name,
            post_aggregation_embedding_name=cfg.chienn.aggregate.post_aggregation_embedding_name,
            dropout=dropout
        )
        self.return_batch = return_batch

    def forward(self, batch: Batch):
        x = super().forward(batch)
        if self.return_batch:
            batch.x = x
            return batch
        else:
            return x

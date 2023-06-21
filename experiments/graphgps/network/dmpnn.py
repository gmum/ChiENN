from argparse import Namespace

import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from submodules.tetra_dmpnn.model.gnn import GNN


@register_network('DMPNN')
class DMPNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, pooling=True):
        super().__init__()
        args = {
            'depth': cfg.gnn.layers,
            'hidden_size': cfg.model.hidden_dim,
            'dropout': cfg.gnn.dropout,
            'gnn_type': 'dmpnn',
            'graph_pool': cfg.model.graph_pooling,
            'tetra': cfg.gnn.tetra.use,
            'message': cfg.gnn.tetra.message,
        }
        self.model = GNN(args=Namespace(**args),
                         num_node_features=cfg.dataset.node_encoder_in_dim,
                         num_edge_features=cfg.dataset.edge_encoder_in_dim,
                         out_dim=dim_out,
                         pooling=pooling)

    def forward(self, batch):
        x = self.model(batch)
        return x, batch.y

import json

import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network

from graphgps.network.utils import get_local_structure_map
from model.alpha_encoder import Encoder
from model.params_interpreter import string_to_object


@register_network('ChIRo')
class ChIRo(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        with open(cfg.model.config_path, 'r') as fp:
            params = json.load(fp)
        for key, value in params['activation_dict'].items():
            params['activation_dict'][key] = string_to_object[value]
        self.encoder = Encoder(F_H_embed=cfg.dataset.node_encoder_in_dim,
                               F_E_embed=cfg.dataset.edge_encoder_in_dim,
                               F_H=cfg.model.hidden_dim,
                               F_H_EConv=cfg.model.hidden_dim,
                               CMP_GAT_N_layers=cfg.gnn.layers,
                               dropout=cfg.gnn.dropout,
                               dim_out=dim_out,
                               **params)

    def forward(self, batch):
        LS_map, alpha_indices = get_local_structure_map(batch.dihedral_angle_index)
        x = self.encoder(batch, LS_map, alpha_indices)
        return x[0], batch.y

import torch
from torch import nn
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder


class GeometricPE(nn.Module):
    """
    Adapted from GeometricTransformer
    """

    def __init__(self, d_model, dims=50):
        super().__init__()
        self.net = torch.nn.Sequential(*[nn.Linear(1, dims,
                                                   bias=True), torch.nn.GELU(), nn.Linear(dims, 1,
                                                                                          bias=True),
                                         torch.nn.GELU()])
        self.embed = nn.Linear(1, d_model,
                               bias=True)

    def forward(self, batch):
        distances_mask = torch.logical_and(batch.mask.unsqueeze(-1), batch.mask.unsqueeze(-2))
        x = self.net(batch.distances.unsqueeze(-1))
        x = torch.masked_fill(x, batch.zero_distances_mask.unsqueeze(-1), 0.0)
        x = torch.masked_fill(x, ~distances_mask.unsqueeze(-1), 0.0)
        x = torch.sum(x, -2)
        x = self.embed(x)
        return x[batch.mask]


@register_node_encoder('GeometricNode')
class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(cfg.dataset.node_encoder_in_dim, emb_dim)
        self.pe = GeometricPE(emb_dim)

    def forward(self, batch):
        batch.x = self.encoder(batch.x) + self.pe(batch)
        return batch

import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.chienn_layer import ChiENN
from graphgps.layer.san_layer import SANLayer
from graphgps.layer.san2_layer import SAN2Layer


@register_network('SANTransformer')
class SANTransformer(torch.nn.Module):
    """Spectral Attention Network (SAN) Graph Transformer.
    https://arxiv.org/abs/2106.03893
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        cfg.gnn.dim_inner = cfg.model.hidden_dim
        cfg.gt.dim_hidden = cfg.model.hidden_dim
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)

        fake_edge_emb = torch.nn.Embedding(1, cfg.gt.dim_hidden)
        # torch.nn.init.xavier_uniform_(fake_edge_emb.weight.data)
        Layer = {
            'SANLayer': SANLayer,
            'SAN2Layer': SAN2Layer,
        }.get(cfg.gt.layer_type)
        layers = []
        for i in range(cfg.gt.layers):
            if cfg.model.add_chienn_layer == 'first':
                add_chienn_layer = i == 0
            elif cfg.model.add_chienn_layer == 'all':
                add_chienn_layer = True
            elif cfg.model.add_chienn_layer == 'none':
                add_chienn_layer = False
            else:
                raise NotImplementedError()
            layers.append(Layer(gamma=cfg.gt.gamma,
                                in_dim=cfg.gt.dim_hidden,
                                out_dim=cfg.gt.dim_hidden,
                                num_heads=cfg.gt.n_heads,
                                full_graph=cfg.gt.full_graph,
                                fake_edge_emb=fake_edge_emb,
                                dropout=cfg.gt.dropout,
                                layer_norm=cfg.gt.layer_norm,
                                batch_norm=cfg.gt.batch_norm,
                                residual=cfg.gt.residual))
            if add_chienn_layer:
                layers.append(ChiENN(
                    cfg.model.hidden_dim,
                    cfg.model.hidden_dim,
                    dropout=0.0,
                    n_heads=cfg.gt.n_heads,
                    return_batch=True))
        self.trf_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
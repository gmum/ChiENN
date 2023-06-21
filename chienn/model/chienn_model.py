from torch import nn, Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from .chienn_layer import ChiENNLayer


class ChiENNModel(nn.Module):
    """
    A simplified version of the ChiENN model used in the experimental part of the paper. To make this implementation
    concise and clear, we excluded computation of RWSE positional encodings. To reproduce the results from the paper,
    use the `experiments` module. Note that this model behaves like GPSModel from experiments/graphgps/network.gps_model.py
    with `local_gnn_type` set to `ChiENN' and `global_model_type` set to `None` (except for the positional encodings).
    Therefore, we wrapped `ChiENNLayer` with `GPSLayer`.
    """

    def __init__(
            self,
            k_neighbors: int = 3,
            in_node_dim: int = 93,
            hidden_dim: int = 128,
            out_dim: int = 1,
            n_layers: int = 3,
            dropout: float = 0.0,
    ):
        """

        Args:
            k_neighbors: number of k consecutive neighbors used to create a chiral-sensitive message. It's `k` from
                the eq. (4) in the paper.
            in_node_dim: number of input node features. Default (93) differs from the value used in the `experiments`
                module (118) as here we explicitly excluded chiral tags, while in the `experiments` we masked them.
            out_dim: output dimension.
            n_layers: number of ChiENN layers.
            dropout: dropout probability.
        """
        super().__init__()
        self.embedding_layer = nn.Linear(in_node_dim, hidden_dim)
        self.gps_layers = nn.ModuleList([
            GPSLayer(
                gnn_layer=ChiENNLayer(
                    hidden_dim=hidden_dim,
                    k_neighbors_embeddings_names=['linear'] * k_neighbors
                ),
                hidden_dim=hidden_dim,
                dropout=dropout,
            ) for _ in range(n_layers)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, out_dim),
        )

    def forward(self, batch: Batch) -> Tensor:
        """
        Run ChiENN model.

        Args:
            batch: a batch representing `batch_size` graphs. Contains the following attributes:
                - x: (num_nodes, hidden_dim) node features
                - batch (num_nodes,): batch indices of the nodes.
                - edge_index (2, num_edges): edge indices
                - circle_index (num_nodes, circle_size): indices of neighbors forming an order around a node.
                - parallel_node_index (num_nodes,): indices of parallel nodes.

        Returns:
            Output of the shape (batch_size, out_dim).
        """
        batch.x = self.embedding_layer(batch.x)
        for gps_layers in self.gps_layers:
            batch.x = gps_layers(batch)
        x, _ = to_dense_batch(batch.x, batch.batch)
        x = x.sum(dim=1)
        return self.output_layer(x)


class GPSLayer(nn.Module):
    """
    A layer that wraps some GNN model and adds a residual connection and a MLP with batch normalization, similarly
    to the GPSLayer from `experiments/graphgps/layer/gps_layer.py`.
    """
    def __init__(
            self,
            gnn_layer: nn.Module,
            hidden_dim: int = 128,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.norm_1 = nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm_2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, batch: Batch) -> Tensor:
        residual_x = batch.x
        x = self.gnn_layer(batch) + residual_x
        x = self.norm_1(x)
        x = self.mlp(x) + x
        return self.norm_2(x)

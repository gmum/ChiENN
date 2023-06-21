import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.graphgym import cfg
from torch_geometric.utils import to_dense_batch


class AtomDistance(nn.Module):
    """
    Adds inversed attoms' distances to a batch.
    """

    def __init__(self):
        super().__init__()
        coords_selection = cfg.model.coords_selection
        if coords_selection == 'start':
            self.coords_selection_fn = lambda x: x[:, :3]
        elif coords_selection == 'end':
            self.coords_selection_fn = lambda x: x[:, -3:]
        elif coords_selection == 'center':
            self.coords_selection_fn = lambda x: (x[:, :3] + x[:, -3:]) * 0.5
        else:
            raise NotImplemented(f'Unknown corrds_selection {coords_selection}.')

    def forward(self, batch: Batch) -> Batch:
        pos = self.coords_selection_fn(batch.pos)
        pos, mask = to_dense_batch(pos, batch.batch)
        distances = torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")
        zero_distances_mask = distances <= 1e-3  # just for safety
        distances = 1. / (distances + 1e-16)
        batch.distances = distances
        batch.zero_distances_mask = zero_distances_mask
        batch.mask = mask
        return batch

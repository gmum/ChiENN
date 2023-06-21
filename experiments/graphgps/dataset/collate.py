from typing import List, Any

from torch_geometric.data.data import BaseData
from torch_geometric.loader.dataloader import Collater

from chienn.data import collate_with_circle_index


class CustomCollater:
    def __init__(self, follow_batch=None, exclude_keys=None, n_neighbors_in_circle=None):
        self.collator = Collater(follow_batch, exclude_keys)
        self.follow_batch = follow_batch
        exclude_keys = exclude_keys if exclude_keys else []
        self.exclude_keys = exclude_keys + ['circle_index']
        self.n_neighbors_in_circle = n_neighbors_in_circle

    def __call__(self, batch: List[Any]):
        elem = batch[0]
        if isinstance(elem, BaseData) and hasattr(elem, 'circle_index'):
            return collate_with_circle_index(batch, self.n_neighbors_in_circle)
        else:
            return self.collator(batch)

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.data.collate import collate
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader.dataloader import Collater
from typing import List, Any


def circle_index_collate(circle_index_list: List[List[List[int]]]) -> Tensor:
    circle_index_tensor_list = []
    for circle_index in circle_index_list:
        n_nodes = len(circle_index_tensor_list)
        circle_index_tensor_list.extend(torch.tensor(circle).long() + n_nodes for circle in circle_index)
    return pad_sequence(circle_index_tensor_list, batch_first=True, padding_value=-1)


class CustomCollater:
    def __init__(self, follow_batch=None, exclude_keys=None, n_neighbors_in_circle=None):
        self.collator = Collater(follow_batch, exclude_keys)
        self.follow_batch = follow_batch
        exclude_keys = exclude_keys if exclude_keys else []
        self.exclude_keys = exclude_keys + ['ccw_circle_index', 'cw_circle_index']
        self.n_neighbors_in_circle = n_neighbors_in_circle

    def _repeat_first_elements(self, circle_index: List[int]) -> List[int]:
        """
        Extends `circle_index` list with its first `self.n_neighbors_in_circle - 1` elements.
        Thanks to this method we don't need to prepare distinct circle_indices for 2- and 3-ary message embeddings.

        Args:
            circle_index: list to be extended.

        Returns:
            Extended list.
        """
        if len(circle_index) == 0:
            return circle_index
        n = len(circle_index) + self.n_neighbors_in_circle - 1
        circle_index = circle_index * self.n_neighbors_in_circle
        return circle_index[:n]

    def _repeat_first_elements_list(self, circle_index_list: List[List[int]]) -> List[List[int]]:
        return [self._repeat_first_elements(circle_index) for circle_index in circle_index_list]

    def __call__(self, batch: List[Any]):
        elem = batch[0]
        if isinstance(elem, BaseData) and hasattr(elem, 'ccw_circle_index'):
            ccw_circle_index = circle_index_collate([self._repeat_first_elements_list(e.ccw_circle_index) for e in batch])
            cw_circle_index = circle_index_collate([self._repeat_first_elements_list(e.cw_circle_index) for e in batch])
            batch = Batch.from_data_list(batch, self.follow_batch,
                                         self.exclude_keys)
            batch.ccw_circle_index = ccw_circle_index
            batch.cw_circle_index = cw_circle_index
            return batch
        else:
            return self.collator(batch)

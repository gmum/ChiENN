from typing import List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data


def collate_circle_index(batch: List[Data], k_neighbors: int) -> Tensor:
    """
    Collates `circle_index` attribute of `Data` objects in `batch` into a single tensor.

    Args:
        batch: a list of `Data` objects.
        k_neighbors: number of incoming neighbors to consider for each node (k in the paper).

    Returns:
        A tensor of shape (num_nodes, circle_size) containing the indices of the (non-parallel) neighbors in the
        pre-computed order. The first (k-1) indices for every atom are repeated, e.g. for k=3, the circle_index[0] may
        be (i_1, i_2, i_3, ..., i_n, i_1, i_2). Therefore, `circle_size` = `max_num_neighbors` + k-1.
    """

    # To simplify the implementation of ChiENNLayer, we extend each `e.circle_index` with its first `k - 1` elements:
    repeated_circle_index = [_repeat_first_elements(e.circle_index, k_neighbors) for e in batch]
    circle_index = _collate_repeated(repeated_circle_index)
    return circle_index


def _collate_repeated(circle_index_list: List[List[List[int]]]) -> Tensor:
    """
    Collates `circle_index_list` into a single tensor.

    Args:
        circle_index_list: a list of `circle_index` lists. Each of `batch_size` lists contains `num_nodes` lists of
            `circle_index` with `circle_size` indices.

    Returns:
        A tensor of shape (total_num_nodes, max_circle_size) containing the indices defining a node order for every node.
    """
    circle_index_tensor_list = []
    for circle_index in circle_index_list:
        n_nodes = len(circle_index_tensor_list)
        circle_index_tensor_list.extend(torch.tensor(circle).long() + n_nodes for circle in circle_index)
    return pad_sequence(circle_index_tensor_list, batch_first=True, padding_value=-1)


def _repeat_first_elements(circle_index_list: List[List[int]], k: int) -> List[List[int]]:
    """
    Extends each `circle_index` from `circle_index_list` with its first `k - 1` elements.
    """
    def _repeat(circle_index: List[int]) -> List[int]:
        """
        Extends `circle_index` list with its first `k - 1` elements.
        """
        if len(circle_index) == 0:
            return circle_index
        n = len(circle_index) + k - 1
        circle_index = circle_index * k
        return circle_index[:n]

    return [_repeat(circle_index) for circle_index in circle_index_list]

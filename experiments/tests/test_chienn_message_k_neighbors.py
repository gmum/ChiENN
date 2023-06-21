import sys

import pytest
from torch import nn

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')
from graphgps.dataset.collate import CustomCollater
from graphgps.dataset.chienn_utils import get_circle_index
from torch_geometric.data import Data

from graphgps.layer.chienn_layer import ChiENNMessageKNeighborsSingleDirection
import torch


@pytest.fixture
def data():
    x = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [-1, -1], [-2, -2], [-3, -3], [-4, -4], [-5, -5]])
    edge_index = torch.tensor(
        [[1, 4], [2, 4], [3, 4], [0, 5], [2, 5], [3, 5], [0, 6], [1, 6], [3, 6], [0, 7], [1, 7], [1, 8]]).T
    pos = torch.tensor(
        [[-1, 1, -1, 0, 0, 0], [-1, -1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, -1, -1, 0, 0, 0],
         [0, 0, 0, -1, 1, -1], [0, 0, 0, -1, -1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, -1, -1], [0, 0, 0, 1, -1, -2]])
    data = Data(x=x, edge_index=edge_index, pos=pos)
    data.ccw_circle_index = get_circle_index(data, ignore_parallel_node=False)
    data.cw_circle_index = get_circle_index(data, clockwise=True, ignore_parallel_node=False)
    return data


def test__chienn_message_two_neighbors_single_direction__works_on_example_k_2(data):
    model = ChiENNMessageKNeighborsSingleDirection(embeddings=[nn.Identity(), nn.Identity()],
                                                   aggregation='concat',
                                                   final_embedding=nn.Identity(),
                                                   mask_by_in_degree=False)
    collater = CustomCollater(n_neighbors_in_circle=2)

    expected_msg = torch.tensor([[[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 [[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 [[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 [[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 [[2, 2, 4, 4],
                                  [4, 4, 3, 3],
                                  [3, 3, 2, 2]],
                                 [[1, 1, 3, 3],
                                  [3, 3, 4, 4],
                                  [4, 4, 1, 1]],
                                 [[1, 1, 4, 4],
                                  [4, 4, 2, 2],
                                  [2, 2, 1, 1]],
                                 [[1, 1, 2, 2],
                                  [2, 2, 1, 1],
                                  [0, 0, 0, 0]],
                                 [[2, 2, 2, 2],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]])

    expected_mask = torch.tensor([[False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, False],
                                  [True, False, False]])
    batch = collater([data])
    msg, mask = model.forward(batch=batch, circle_index=batch.ccw_circle_index)

    assert torch.equal(expected_msg, msg)
    assert torch.equal(expected_mask, mask)


def test__chienn_message_two_neighbors_single_direction__works_on_example_k_2_mask_repeated(data):
    model = ChiENNMessageKNeighborsSingleDirection(embeddings=[nn.Identity(), nn.Identity()],
                                                   aggregation='concat',
                                                   final_embedding=nn.Identity(),
                                                   mask_by_in_degree=True)
    collater = CustomCollater(n_neighbors_in_circle=2)

    expected_msg = torch.tensor([[[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 [[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 [[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 [[0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]],
                                 [[2, 2, 4, 4],
                                  [4, 4, 3, 3],
                                  [3, 3, 2, 2]],
                                 [[1, 1, 3, 3],
                                  [3, 3, 4, 4],
                                  [4, 4, 1, 1]],
                                 [[1, 1, 4, 4],
                                  [4, 4, 2, 2],
                                  [2, 2, 1, 1]],
                                 [[1, 1, 2, 2],
                                  [2, 2, 1, 1],
                                  [0, 0, 0, 0]],
                                 [[2, 2, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]]])

    expected_mask = torch.tensor([[False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, False],
                                  [True, False, False]])
    batch = collater([data])
    msg, mask = model.forward(batch=batch, circle_index=batch.ccw_circle_index)

    assert torch.equal(expected_msg, msg)
    assert torch.equal(expected_mask, mask)


def test__chienn_message_two_neighbors_single_direction__works_on_example_k_3(data):
    model = ChiENNMessageKNeighborsSingleDirection(embeddings=[nn.Identity(), nn.Identity(), nn.Identity()],
                                                   aggregation='concat',
                                                   final_embedding=nn.Identity(),
                                                   mask_by_in_degree=False)
    collater = CustomCollater(n_neighbors_in_circle=3)

    expected_msg = torch.tensor([[[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[2, 2, 4, 4, 3, 3],
                                  [4, 4, 3, 3, 2, 2],
                                  [3, 3, 2, 2, 4, 4]],

                                 [[1, 1, 3, 3, 4, 4],
                                  [3, 3, 4, 4, 1, 1],
                                  [4, 4, 1, 1, 3, 3]],

                                 [[1, 1, 4, 4, 2, 2],
                                  [4, 4, 2, 2, 1, 1],
                                  [2, 2, 1, 1, 4, 4]],

                                 [[1, 1, 2, 2, 1, 1],
                                  [2, 2, 1, 1, 2, 2],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[2, 2, 2, 2, 2, 2],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]]])

    expected_mask = torch.tensor([[False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, False],
                                  [True, False, False]])

    batch = collater([data])
    msg, mask = model.forward(batch=batch, circle_index=batch.ccw_circle_index)

    assert torch.equal(expected_msg, msg)
    assert torch.equal(expected_mask, mask)


def test__chienn_message_two_neighbors_single_direction__works_on_example_k_3_mask_repeated(data):
    model = ChiENNMessageKNeighborsSingleDirection(embeddings=[nn.Identity(), nn.Identity(), nn.Identity()],
                                                   aggregation='concat',
                                                   final_embedding=nn.Identity(),
                                                   mask_by_in_degree=True)
    collater = CustomCollater(n_neighbors_in_circle=3)

    expected_msg = torch.tensor([[[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[2, 2, 4, 4, 3, 3],
                                  [4, 4, 3, 3, 2, 2],
                                  [3, 3, 2, 2, 4, 4]],

                                 [[1, 1, 3, 3, 4, 4],
                                  [3, 3, 4, 4, 1, 1],
                                  [4, 4, 1, 1, 3, 3]],

                                 [[1, 1, 4, 4, 2, 2],
                                  [4, 4, 2, 2, 1, 1],
                                  [2, 2, 1, 1, 4, 4]],

                                 [[1, 1, 2, 2, 0, 0],
                                  [2, 2, 1, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 0]],

                                 [[2, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]]])

    expected_mask = torch.tensor([[False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [False, False, False],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, False],
                                  [True, False, False]])

    batch = collater([data])
    msg, mask = model.forward(batch=batch, circle_index=batch.ccw_circle_index)

    assert torch.equal(expected_msg, msg)
    assert torch.equal(expected_mask, mask)

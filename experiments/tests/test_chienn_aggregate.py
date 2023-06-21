import sys

import pytest

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')
from graphgps.dataset.chienn_utils import get_circle_index
from torch_geometric.data import Data
from graphgps.dataset.collate import CustomCollater

from graphgps.layer.chienn_layer import ChiENNMessageKNeighbors, \
    ChiENNAggregate
import torch


@pytest.fixture
def data():
    x = torch.tensor([[1], [2], [3], [4], [-1], [-2], [-3], [-4]])
    edge_index = torch.tensor(
        [[1, 4], [2, 4], [3, 4], [0, 5], [2, 5], [3, 5], [0, 6], [1, 6], [3, 6], [0, 7], [1, 7]]).T
    pos = torch.tensor(
        [[-1, 1, -1, 0, 0, 0], [-1, -1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, -1, -1, 0, 0, 0],
         [0, 0, 0, -1, 1, -1], [0, 0, 0, -1, -1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, -1, -1]])
    data = Data(x=x.float(), edge_index=edge_index, pos=pos.float())
    data.ccw_circle_index = get_circle_index(data, ignore_parallel_node=False)
    data.cw_circle_index = get_circle_index(data, clockwise=True, ignore_parallel_node=False)
    return CustomCollater(n_neighbors_in_circle=2)([data])


def test__chienn_aggregate__works_on_example(data):
    message_model = ChiENNMessageKNeighbors(in_dim=1,
                                            out_dim=2,
                                            embedding_names=['identity', 'identity'],
                                            aggregation='concat',
                                            final_embedding_name='identity',
                                            single_direction=False,
                                            shared_weights=False,
                                            mask_by_in_degree=False)

    aggregate_model = ChiENNAggregate(x_dim=1,
                                      msg_dim=2,
                                      self_embedding_name='self_concat',
                                      aggregation='mean',
                                      embedding_name='identity',
                                      final_embedding_name='identity',
                                      parallel_embedding_name='none')

    ccw_msg, ccw_mask, cw_msg, cw_mask = message_model.forward(data)
    result = aggregate_model.forward(data, ccw_msg, ccw_mask, cw_msg, cw_mask)

    expected_result = torch.tensor([[1.0000000000, 1.0000000000],
                                    [2.0000000000, 2.0000000000],
                                    [3.0000000000, 3.0000000000],
                                    [4.0000000000, 4.0000000000],
                                    [2.4285714626, 2.4285714626],
                                    [2.0000000000, 2.0000000000],
                                    [1.5714285374, 1.5714285374],
                                    [0.4000000060, 0.4000000060]])

    assert torch.allclose(result, expected_result)


def test__chienn_aggregate__works_on_example_with_parallel(data):
    message_model = ChiENNMessageKNeighbors(in_dim=1,
                                            out_dim=2,
                                            embedding_names=['identity', 'identity'],
                                            aggregation='concat',
                                            final_embedding_name='identity',
                                            single_direction=False,
                                            shared_weights=False,
                                            mask_by_in_degree=False)

    aggregate_model = ChiENNAggregate(x_dim=1,
                                      msg_dim=2,
                                      self_embedding_name='self_concat',
                                      aggregation='mean',
                                      embedding_name='identity',
                                      final_embedding_name='identity',
                                      parallel_embedding_name='self_concat')

    data.parallel_node_index = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0])
    ccw_msg, ccw_mask, cw_msg, cw_mask = message_model.forward(data)
    result = aggregate_model.forward(data, ccw_msg, ccw_mask, cw_msg, cw_mask)

    expected_result = torch.tensor([[1.5000000000, 1.5000000000],
                                    [2.5000000000, 2.5000000000],
                                    [3.5000000000, 3.5000000000],
                                    [1.5000000000, 1.5000000000],
                                    [1.8750000000, 1.8750000000],
                                    [1.3750000000, 1.3750000000],
                                    [0.8750000000, 0.8750000000],
                                    [0.5000000000, 0.5000000000]])

    assert torch.allclose(result, expected_result)


@pytest.mark.parametrize('out_dim, n_heads', [(12, 3), (18, 2)])
def test__chienn_aggregate__multihead_attention_returns_proper_shape(data, out_dim, n_heads):
    message_model = ChiENNMessageKNeighbors(in_dim=1,
                                            out_dim=out_dim,
                                            embedding_names=['linear', 'linear'],
                                            aggregation='add',
                                            final_embedding_name='identity',
                                            single_direction=False,
                                            shared_weights=False,
                                            mask_by_in_degree=False)

    aggregate_model = ChiENNAggregate(x_dim=1,
                                      msg_dim=out_dim,
                                      self_embedding_name='linear',
                                      aggregation='scale_dot_attention',
                                      embedding_name='identity',
                                      n_heads=n_heads,
                                      final_embedding_name='ELU',
                                      parallel_embedding_name='none')

    ccw_msg, ccw_mask, cw_msg, cw_mask = message_model.forward(data)
    result = aggregate_model.forward(data, ccw_msg, ccw_mask, cw_msg, cw_mask)

    assert result.shape == (data.x.shape[0], out_dim)

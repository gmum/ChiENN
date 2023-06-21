import sys

from torch_geometric.data import Batch

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')
import torch
import torch_geometric.data
import pytest

from graphgps.dataset.utils import Molecule3DEmbedder, get_chiro_data_from_mol, to_edge_graph


def test__to_edge_graph_works_with_example():
    x = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
    pos = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 3, 2], [3, 3, 3]])
    edge_index = torch.tensor([[0, 2], [1, 2], [2, 3]]).T
    edge_attr = torch.tensor([[0], [1], [2]])
    data = torch_geometric.data.Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    expected_x = torch.tensor(
        [[0, 0, 0, 2, 2], [1, 1, 1, 2, 2], [2, 2, 0, 0, 0], [2, 2, 1, 1, 1], [2, 2, 2, 3, 3], [3, 3, 2, 2, 2]])
    expected_pos = torch.tensor([[0, 0, 0, 2, 3, 2], [1, 1, 1, 2, 3, 2],
                                 [2, 3, 2, 0, 0, 0], [2, 3, 2, 1, 1, 1],
                                 [2, 3, 2, 3, 3, 3], [3, 3, 3, 2, 3, 2]])

    expected_edge_index = torch.tensor(
        [[2, 0], [3, 1], [0, 2], [1, 2], [5, 2], [0, 3], [1, 3], [5, 3], [0, 4], [1, 4], [5, 4], [4, 5]]).T
    expected_edge_attr = torch.tensor([[0, 0, 0, 0],
                                       [1, 1, 1, 1],
                                       [0, 2, 2, 0],
                                       [1, 2, 2, 0],
                                       [2, 2, 2, 0],
                                       [0, 2, 2, 1],
                                       [1, 2, 2, 1],
                                       [2, 2, 2, 1],
                                       [0, 2, 2, 2],
                                       [1, 2, 2, 2],
                                       [2, 2, 2, 2],
                                       [2, 3, 3, 2]])
    expected_parallel_node_index = torch.tensor([2, 3, 0, 1, 5, 4])

    result = to_edge_graph(data)
    print(result.edge_attr)
    assert torch.equal(result.x, expected_x)
    assert torch.equal(result.pos, expected_pos)
    assert torch.equal(result.edge_index, expected_edge_index)
    assert torch.equal(result.edge_attr, expected_edge_attr)
    assert torch.equal(result.parallel_node_index, expected_parallel_node_index)

@pytest.mark.parametrize('batch_size', [2, 16])
def test__to_edge_graph_parallel_node_index_collates_correctly(batch_size: int):
    x = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
    pos = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 3, 2], [3, 3, 3]])
    edge_index = torch.tensor([[0, 2], [1, 2], [2, 3]]).T
    edge_attr = torch.tensor([[0], [1], [2]])
    data = torch_geometric.data.Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    edge_data = to_edge_graph(data)
    single_parallel_node_index = edge_data.parallel_node_index
    n = edge_data.x.shape[0]
    expected_parallel_node_index = torch.cat([single_parallel_node_index + i * n for i in range(batch_size)])
    result = Batch.from_data_list([edge_data] * batch_size)

    assert torch.equal(result.parallel_node_index, expected_parallel_node_index)


@pytest.mark.parametrize('smiles', ['CCC', 'CCCCC', 'CCCCCC'])
def test__to_edge_graph_works_with_molecules(smiles: str):
    embedder = Molecule3DEmbedder(max_number_of_atoms=100)
    mol = embedder.embed(smiles)
    data = get_chiro_data_from_mol(mol)
    n_edges = data.edge_index.shape[-1]
    node_dim = data.x.shape[-1]
    edge_dim = data.edge_attr.shape[-1]

    result = to_edge_graph(data)

    assert result.x.shape == (n_edges, node_dim + edge_dim + node_dim)
    assert result.parallel_node_index.shape == (n_edges,)
    assert result.edge_index.shape[0] == 2
    n_edges = result.edge_index.shape[-1]
    assert result.edge_attr.shape == (n_edges, edge_dim + node_dim + edge_dim)

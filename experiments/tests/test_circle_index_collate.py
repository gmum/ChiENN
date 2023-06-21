import sys

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')
from graphgps.dataset.dataloader import CustomDataLoader
import torch
from typing import List, Tuple

from torch import Tensor

from graphgps.dataset.collate import circle_index_collate

import pytest

from graphgps.dataset.utils import Molecule3DEmbedder, get_chiro_data_from_mol, to_edge_graph


@pytest.mark.parametrize('circle_index_list, expected_result', [
    ([
         [[], [0, 2, 0], [], [1, 3, 2, 1]],
         [[1, 3, 2, 1], []],
         [[0, 2, 0], []],
         [[], [0, 2, 0], []]
     ], torch.tensor([
        [-1, -1, -1, -1],
        [0, 2, 0, -1],
        [-1, -1, -1, -1],
        [1, 3, 2, 1],
        [5, 7, 6, 5],
        [-1, -1, -1, -1],
        [6, 8, 6, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [8, 10, 8, -1],
        [-1, -1, -1, -1],
    ])),
])
def test__circle_index_collate__works_correctly(circle_index_list: List[List[int]], expected_result: Tensor):
    result = circle_index_collate(circle_index_list)
    assert torch.equal(result, expected_result)


@pytest.mark.parametrize('smiles, batch_size, expected_shape', [('CC', 1, (14, 4)), ('CC', 3, (3 * 14, 4))])
def test__circle_index__batches_correctly(smiles: str, batch_size: int, expected_shape: Tuple[int, int]):
    embedder = Molecule3DEmbedder(max_number_of_atoms=100)
    smiles_list = [smiles] * batch_size
    mol_list = [embedder.embed(smiles) for smiles in smiles_list]
    data_list = [get_chiro_data_from_mol(mol) for mol in mol_list]
    data_list = [to_edge_graph(data) for data in data_list]

    loader = CustomDataLoader(data_list, batch_size=batch_size, n_neighbors_in_circle=2)
    batch = next(iter(loader))

    assert batch.ccw_circle_index.shape == batch.cw_circle_index.shape == expected_shape
    assert not torch.equal(batch.ccw_circle_index, batch.cw_circle_index)

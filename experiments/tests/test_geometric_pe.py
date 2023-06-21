import random
import sys

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')
from torch_geometric.data import Data, Batch

from graphgps.encoder.geometric_node_encoder import GeometricPE
from graphgps.layer.utils import AtomDistance

import torch
import pytest


@pytest.mark.parametrize('d_model, n_mols', [(16, 10), (128, 3), (3, 3)])
def test__geometric_transformer_pe__returns_correct_shape(d_model, n_mols):
    torch.manual_seed(0)
    random.seed(0)
    dist = AtomDistance()
    pe = GeometricPE(d_model)

    data_list = []
    for i in range(n_mols):
        n_nodes = random.choice(range(2, 20))
        data = Data(x=torch.randn((n_nodes, d_model)), pos=torch.randn((n_nodes, 3)))
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    batch = dist(batch)
    output = pe(batch)

    assert output.shape == batch.x.shape

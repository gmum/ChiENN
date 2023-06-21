import sys

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')
from graphgps.dataset.collate import CustomCollater
from graphgps.network.dmpnn import DMPNN
import pytest
import torch
from torch_geometric.data import Data, Batch

from torch_geometric.graphgym import cfg, set_cfg, FeatureEncoder

from graphgps.dataset.utils import Molecule3DEmbedder, get_chiro_data_from_mol, to_edge_graph, mask_chiral_edge_graph, \
    mask_chiral_default


def get_data(smiles: str) -> Data:
    embedder = Molecule3DEmbedder(max_number_of_atoms=100)
    mol = embedder.embed(smiles)
    data = get_chiro_data_from_mol(mol)
    data.parity_atoms = data.x[:, -4] -  data.x[:, -3]
    return data


BASE_CONFIG_PATH = 'configs/models/Tetra_DMPNN/Tetra_DMPNN.yaml'


@pytest.mark.parametrize(
    'smiles_1, smiles_2',
    [
        ('C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O'),
        ('BrC1=C[C@@H](c2ccc(-c3ccccc3)cc2)CC(c2nc(-c3ccccc3)nc(-c3ccccc3)n2)=C1',
         'BrC1=C[C@H](c2ccc(-c3ccccc3)cc2)CC(c2nc(-c3ccccc3)nc(-c3ccccc3)n2)=C1'),
        ('Br[C@@H]1SC=CN1C1CC1', 'Br[C@H]1SC=CN1C1CC1'),

    ])
def test_tetra__distinguish_enantiomers(smiles_1: str, smiles_2: str):
    set_cfg(cfg)
    cfg.merge_from_file(BASE_CONFIG_PATH)
    cfg.gnn.dim_inner = cfg.model.hidden_dim
    encoder = FeatureEncoder(cfg.dataset.node_encoder_in_dim)
    model = DMPNN(dim_in=cfg.model.hidden_dim, dim_out=cfg.model.hidden_dim, pooling=False)
    model.eval()

    data_list = [get_data(smiles_1), get_data(smiles_2)]
    data_list = [mask_chiral_default(data) for data in data_list]

    assert torch.allclose(data_list[0].x, data_list[1].x)

    batch = CustomCollater()(data_list)

    batch = encoder(batch)
    result_x = model(batch)[0]
    n_nodes = len(batch.x)
    assert not torch.allclose(result_x[n_nodes // 2:], result_x[:n_nodes // 2], rtol=1e-6, atol=1e-6)

import sys

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')
from grid_search import update_cfg
from graphgps.dataset.collate import CustomCollater

import pytest
import torch
from torch_geometric.data import Data
from typing import Dict, Any

from torch_geometric.graphgym import cfg, set_cfg, FeatureEncoder

from graphgps.dataset.utils import Molecule3DEmbedder, get_chiro_data_from_mol, to_edge_graph, mask_chiral_edge_graph
from graphgps.layer.chienn_layer import ChiENN


def get_data(smiles: str) -> Data:
    embedder = Molecule3DEmbedder(max_number_of_atoms=100)
    mol = embedder.embed(smiles)
    data = get_chiro_data_from_mol(mol)
    data = to_edge_graph(data)
    return data


BASE_CONFIG_PATH = 'configs/models/ChiENN/ChiENN.yaml'


@pytest.mark.parametrize(
    'smiles_1, smiles_2, settings, distinguish',
    [
        ('C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.message.embedding_names': ['linear', 'linear'],
             'chienn.aggregate.aggregation': 'mean'
         }, False),
        ('C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.aggregate.aggregation': 'gat_attention',
             'chienn.message.embedding_names': ['linear', 'linear'],
             'chienn.aggregate.embedding_name': 'linear'
         }, True),
        ('BrC1=C[C@@H](c2ccc(-c3ccccc3)cc2)CC(c2nc(-c3ccccc3)nc(-c3ccccc3)n2)=C1',
         'BrC1=C[C@H](c2ccc(-c3ccccc3)cc2)CC(c2nc(-c3ccccc3)nc(-c3ccccc3)n2)=C1',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.aggregate.aggregation': 'gat_attention',
             'chienn.message.embedding_names': ['linear', 'linear'],
             'chienn.aggregate.embedding_name': 'linear'
         }, True),
        ('Br[C@@H]1SC=CN1C1CC1', 'Br[C@H]1SC=CN1C1CC1',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.message.embedding_names': ['linear', 'linear'],
             'chienn.aggregate.aggregation': 'gat_attention',
             'chienn.aggregate.embedding_name': 'linear'
         }, True),
        ('C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.message.aggregation': 'minus',
             'chienn.message.embedding_names': ['double', 'identity'],
             'chienn.aggregate.aggregation': 'gat_attention',
             'chienn.aggregate.embedding_name': 'linear',
             'chienn.aggregate.self_embedding_name': 'linear',
             'chienn.message.single_direction': True
         }, True),
        ('C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.message.aggregation': 'minus',
             'chienn.message.embedding_names': ['double', 'identity'],
             'chienn.aggregate.aggregation': 'gat_attention',
             'chienn.aggregate.embedding_name': 'linear',
             'chienn.aggregate.self_embedding_name': 'linear',
             'chienn.message.single_direction': False
         }, False),
        ('C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.aggregate.aggregation': 'scale_dot_attention',
             'chienn.message.embedding_names': ['lienar', 'linear'],
             'chienn.aggregate.embedding_name': 'linear'
         }, True),
        ('C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.aggregate.aggregation': 'gat_attention',
             'chienn.message.embedding_names': ['lienar', 'linear'],
             'chienn.aggregate.embedding_name': 'identity'
         }, True),
        ('C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.message.aggregation': 'mul',
             'chienn.message.embedding_names': ['lienar', 'linear'],
             'chienn.message.first_embedding_name': 'scalar+Sigmoid',
             'chienn.aggregate.aggregation': 'mean',
             'chienn.aggregate.embedding_name': 'identity'
         }, True),
        ('C[C@H](C(=O)O)O', 'C[C@@H](C(=O)O)O',
         {
             'dataset.node_encoder_name': 'LinearNode',
             'chienn.aggregate.aggregation': 'scale_dot_attention',
             'chienn.message.embedding_names': ['lienar', 'linear'],
             'chienn.aggregate.embedding_name': 'identity'
         }, True),
        ('CC(C(=O)O)O', 'CC(C(=O)O)O', {'dataset.node_encoder_name': 'LinearNode'}, False)
    ])
def test_chienn__distinguish_enantiomers(smiles_1: str, smiles_2: str, settings: Dict[str, Any], distinguish: bool):
    set_cfg(cfg)
    cfg.merge_from_file(BASE_CONFIG_PATH)
    update_cfg(cfg, settings)
    cfg.gnn.dim_inner = cfg.model.hidden_dim
    encoder = FeatureEncoder(cfg.dataset.node_encoder_in_dim)
    model = ChiENN(in_dim=cfg.model.hidden_dim, out_dim=cfg.model.hidden_dim, dropout=0.0, n_heads=4)

    data_list = [get_data(smiles_1), get_data(smiles_2)]
    data_list = [mask_chiral_edge_graph(data) for data in data_list]

    assert torch.allclose(data_list[0].x, data_list[1].x)

    batch = CustomCollater(n_neighbors_in_circle=2)(data_list)
    batch = mask_chiral_edge_graph(batch)

    batch = encoder(batch)
    result_x = model(batch)

    n_nodes = len(batch.x)
    if distinguish:
        assert not torch.allclose(result_x[n_nodes // 2:], result_x[:n_nodes // 2], rtol=1e-6, atol=1e-6)
    else:
        assert torch.allclose(result_x[n_nodes // 2:], result_x[:n_nodes // 2], rtol=1e-6, atol=1e-6)

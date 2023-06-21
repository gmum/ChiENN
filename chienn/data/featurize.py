from typing import List

from torch_geometric.data import Data, Batch

from chienn.data.edge_graph.collate_circle_index import collate_circle_index
from chienn.data.edge_graph.to_edge_graph import to_edge_graph
from chienn.data.featurization.mol_to_data import mol_to_data
from chienn.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol


def smiles_to_data_with_circle_index(smiles: str) -> Data:
    """
    Transforms a SMILES string into a torch_geometric Data object that can be fed into the ChiENNLayer.
    Args:
        smiles: a SMILES string.

    Returns:
        Data object containing the following attributes:
            - x (num_nodes,): node features.
            - edge_index (2, num_edges): edge index.
            - circle_index (num_nodes, circle_size): neighbors indices ordered around a node.
    """
    mol = smiles_to_3d_mol(smiles)
    data = mol_to_data(mol)
    data = to_edge_graph(data)
    data.pos = None
    return data


def collate_with_circle_index(data_list: List[Data], k_neighbors: int) -> Batch:
    """
    Collates a list of Data objects into a Batch object.

    Args:
        data_list: a list of Data objects. Each Data object must contain `circle_index` attribute.
        k_neighbors: number of k consecutive neighbors to be used in the message passing step.

    Returns:
        Batch object containing the collate attributes from data objects, including `circle_index` collated
        to shape (total_num_nodes, max_circle_size).
    """
    batch = Batch.from_data_list(data_list, exclude_keys=['circle_index'])
    batch.circle_index = collate_circle_index(data_list, k_neighbors)
    return batch

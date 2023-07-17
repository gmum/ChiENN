# Adapted from https://github.com/keiradams/ChIRo/blob/main/model/embedding_functions.py

import numpy as np
import rdkit
import torch
from rdkit.Chem import rdMolTransforms, Mol, Atom, Bond
from torch_geometric.data import Data
from typing import List

atom_types = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
formal_charges = [-1, -2, 1, 2, 0]
degree = [0, 1, 2, 3, 4, 5, 6]
num_hs = [0, 1, 2, 3, 4]
local_chiral_tags = [0, 1, 2, 3]
hybridization = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
]
bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']


def mol_to_data(mol: Mol) -> Data:
    """
    Transforms a rdkit mol object into a torch_geometric Data object.
    Args:
        mol: rdKit mol object.

    Returns:
        Data object containing the following attributes:
            - x: node features.
            - edge_index: edge index.
            - edge_attr: edge features.
            - pos: node positions.
    """
    # Edge Index
    adj = rdkit.Chem.GetAdjacencyMatrix(mol)
    edge_index = adjacency_to_undirected_edge_index(adj)

    # Edge Features
    bonds = []
    for b in range(int(edge_index.shape[1] / 2)):
        bond_index = edge_index[:, ::2][:, b]
        bond = mol.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = get_edge_features(bonds)

    # Node Features
    atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
    node_features = get_node_features(atoms)

    # Positions
    pos = get_positions(mol)

    # Create Data object
    data = Data(x=torch.as_tensor(node_features),
                edge_index=torch.as_tensor(edge_index).long(),
                edge_attr=torch.as_tensor(edge_features),
                pos=torch.as_tensor(pos).float())

    return data


def one_hot_embedding(value: int, options: List[int]) -> List[int]:
    """
    Encodes a value into a one-hot embedding.
    Args:
        value: a value which index will be retrieved from options and encoded.
        options: a list of possible values.

    Returns:
        One-hot embedding of the value.
    """
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding


def adjacency_to_undirected_edge_index(adj: np.ndarray) -> np.ndarray:
    """
    Converts an adjacency matrix into an edge index.
    Args:
        adj: adjacency matrix.

    Returns:
        Edge index.
    """
    adj = np.triu(np.array(adj, dtype=int))  # keeping just upper triangular entries from sym matrix
    array_adj = np.array(np.nonzero(adj), dtype=int)  # indices of non-zero values in adj matrix
    edge_index = np.zeros((2, 2 * array_adj.shape[1]), dtype=int)  # placeholder for undirected edge list
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index


def get_node_features(atoms: List[Atom]) -> np.ndarray:
    """
    Gets an array of node features from a list of atoms.
    Args:
        atoms: list of atoms of shape (N).

    Returns:
        Array of node features of shape (N, 43).
    """
    num_features = (len(atom_types) + 1) + \
                   (len(degree) + 1) + \
                   (len(formal_charges) + 1) + \
                   (len(num_hs) + 1) + \
                   (len(hybridization) + 1) + \
                   2  # 43

    node_features = np.zeros((len(atoms), num_features))
    for node_index, node in enumerate(atoms):
        features = one_hot_embedding(node.GetSymbol(), atom_types)  # atom symbol, dim=12 + 1
        features += one_hot_embedding(node.GetTotalDegree(), degree)  # total number of bonds, H included, dim=7 + 1
        features += one_hot_embedding(node.GetFormalCharge(), formal_charges)  # formal charge, dim=5+1
        features += one_hot_embedding(node.GetTotalNumHs(), num_hs)  # total number of bonded hydrogens, dim=5 + 1
        features += one_hot_embedding(node.GetHybridization(), hybridization)  # hybridization state, dim=7 + 1
        features += [int(node.GetIsAromatic())]  # whether atom is part of aromatic system, dim = 1
        features += [node.GetMass() * 0.01]  # atomic mass / 100, dim=1
        node_features[node_index, :] = features

    return np.array(node_features, dtype=np.float32)


def get_edge_features(bonds: List[Bond]) -> np.ndarray:
    """
    Gets an array of edge features from a list of bonds.
    Args:
        bonds: a list of bonds of shape (N).

    Returns:
        Array of edge features of shape (N, 7).
    """
    num_features = (len(bond_types) + 1) + 2  # 7

    edge_features = np.zeros((len(bonds) * 2, num_features))
    for edge_index, edge in enumerate(bonds):
        features = one_hot_embedding(str(edge.GetBondType()), bond_types)  # dim=4+1
        features += [int(edge.GetIsConjugated())]  # dim=1
        features += [int(edge.IsInRing())]  # dim=1

        # Encode both directed edges to get undirected edge
        edge_features[2 * edge_index: 2 * edge_index + 2, :] = features

    return np.array(edge_features, dtype=np.float32)


def get_positions(mol: rdkit.Chem.Mol) -> np.ndarray:
    """
    Gets the 3D positions of the atoms in the molecule.
    Args:
        mol: a molecule embedded in 3D space with N atoms.

    Returns:
        Array of positions of shape (N, 3).
    """
    conf = mol.GetConformer()
    return np.array(
        [
            [
                conf.GetAtomPosition(k).x,
                conf.GetAtomPosition(k).y,
                conf.GetAtomPosition(k).z,
            ]
            for k in range(mol.GetNumAtoms())
        ]
    )

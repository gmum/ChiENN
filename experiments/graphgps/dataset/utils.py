import ssl
import sys
import urllib
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rdkit.Chem
import torch
import torch_geometric
import torch_geometric.data
from rdkit import Chem
from torch import Tensor
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.loader import create_dataset

from chienn.data.edge_graph import to_edge_graph
from chienn.data.featurization import smiles_to_3d_mol
from graphgps.dataset.dataloader import CustomDataLoader
from model.datasets_samplers import SingleConformerBatchSampler
from model.embedding_functions import embedConformerWithAllPaths


def download_url_to_path(url: str, path: str, log: bool = True):
    """
    Analogous to torch_geometric.graphgym.download_url, but allows to download file to a specific path.
    Well... there is a lot of phds in torch_geometric team.
    """
    path = Path(path)
    if path.exists():  # pragma: no cover
        if log:
            print(f'Using existing file {path}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    path.parent.mkdir(parents=True, exist_ok=True)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def get_positions(mol: rdkit.Chem.Mol):
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


def get_chiro_data_from_mol(mol: rdkit.Chem.Mol):
    """
    Copied from `ChIRo.model.datasets_samplers.MaskedGraphDataset.process_mol`. It encoded molecule with some basic
    chemical features. It also provides chiral tag, which can be then masked in `graphgps.dataset.rs_dataset.RS`.
    """
    atom_symbols, edge_index, edge_features, node_features, bond_distances, bond_distance_index, bond_angles, bond_angle_index, dihedral_angles, dihedral_angle_index = embedConformerWithAllPaths(
        mol, repeats=False)

    bond_angles = bond_angles % (2 * np.pi)
    dihedral_angles = dihedral_angles % (2 * np.pi)
    pos = get_positions(mol)

    data = torch_geometric.data.Data(x=torch.as_tensor(node_features),
                                     edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                                     edge_attr=torch.as_tensor(edge_features),
                                     pos=torch.as_tensor(pos, dtype=torch.float))
    data.bond_distances = torch.as_tensor(bond_distances)
    data.bond_distance_index = torch.as_tensor(bond_distance_index, dtype=torch.long).T
    data.bond_angles = torch.as_tensor(bond_angles)
    data.bond_angle_index = torch.as_tensor(bond_angle_index, dtype=torch.long).T
    data.dihedral_angles = torch.as_tensor(dihedral_angles)
    data.dihedral_angle_index = torch.as_tensor(dihedral_angle_index, dtype=torch.long).T

    return data


def get_custom_loader(dataset, sampler, batch_size, shuffle=True, dataframe=None):
    """
    Returns DataLoader with sampler from ChIRo repository. For each enantiomer in the batch, it randomly
    samples a conformer for that enantiomer and a conformer for its opposite enantiomer with the same 2D graph.
    """
    n_neighbors_in_circle = len(cfg.chienn.message.k_neighbors_embeddings_names)
    if sampler == 'single_conformer_sampler':
        single_conformer_dataframe = dataframe.groupby('ID').sample(1)
        sampler = SingleConformerBatchSampler(single_conformer_dataframe,
                                              dataframe,
                                              batch_size,
                                              N_pos=0,
                                              N_neg=1,
                                              withoutReplacement=True,
                                              stratified=True)
        return CustomDataLoader(dataset, batch_sampler=sampler, num_workers=cfg.num_workers,
                                n_neighbors_in_circle=n_neighbors_in_circle)
    elif sampler == "full_batch" or len(dataset) > 1:
        return CustomDataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=cfg.num_workers,
                                pin_memory=True, n_neighbors_in_circle=n_neighbors_in_circle)
    else:
        raise NotImplementedError()


def create_custom_loader():
    """
    Analogous to dummy create_loader function from torch_geometric.graphgym. It uses a list of dataframes in a dataset
    if provided and returns it at the end. Indeed, this is a hack.
    """
    dataset = create_dataset()
    train_df, val_df, test_df = getattr(dataset, 'dataframes', [None, None, None])

    train_id = dataset.data['train_graph_index']
    train_loader = get_custom_loader(dataset[train_id], cfg.train.sampler, cfg.train.batch_size,
                                     shuffle=True, dataframe=train_df)
    delattr(dataset.data, 'train_graph_index')

    val_id = dataset.data['val_graph_index']
    val_loader = get_custom_loader(dataset[val_id], cfg.val.sampler, cfg.train.batch_size,
                                   shuffle=False, dataframe=val_df)
    delattr(dataset.data, 'val_graph_index')

    test_id = dataset.data['test_graph_index']
    test_loader = get_custom_loader(dataset[test_id], cfg.test.sampler, cfg.train.batch_size,
                                    shuffle=False, dataframe=test_df)
    delattr(dataset.data, 'test_graph_index')

    return [train_loader, val_loader, test_loader], [train_df, val_df, test_df]


def convert_target_for_task(target: Tensor, task_type: str, scale_label: float = 1.0) -> Tensor:
    if task_type in ['regression', 'regression_rank']:
        return target.float() * scale_label
    elif task_type == 'classification_multilabel':
        return target.float().view(1, -1)
    return target.long()


def get_ranking_accuracies(results_df, mode='>=', difference_threshold=0.001):
    """
    Copied from ChIRo.experiment_analysis.
    Args:
        results_df: dataframe
        mode: described in the paper :)
        difference_threshold: minimum difference between binding scores between enantiomers that will be treated
            as significant distinctive difference.

    Returns:
        Bunch of useless staff.
    """
    stats = results_df.groupby("ID")["outputs"].agg([np.mean, np.std]).merge(results_df, on='ID').reset_index(drop=True)

    smiles_groups_std = results_df.groupby(['ID', 'SMILES_nostereo'])['targets', 'outputs'].std().reset_index()
    smiles_groups_mean = results_df.groupby(['ID', 'SMILES_nostereo'])['targets', 'outputs'].mean().reset_index()
    smiles_groups_count = results_df.groupby(['ID', 'SMILES_nostereo'])['targets', 'outputs'].count().reset_index()

    stereoisomers_df = deepcopy(smiles_groups_mean).rename(columns={'outputs': 'mean_predicted_score'})
    stereoisomers_df['std_predicted_score'] = smiles_groups_std['outputs']
    stereoisomers_df['count'] = smiles_groups_count.targets  # score here simply contains the count

    stereoisomers_df_margins = stereoisomers_df.merge(
        pd.DataFrame(stereoisomers_df.groupby('SMILES_nostereo').apply(lambda x: np.max(x.targets) - np.min(x.targets)),
                     columns=['difference']), on='SMILES_nostereo')
    top_1_margins = []
    margins = np.arange(0.1, 2.1, 0.1)  # originally it was ranging from 0.3 with no particular reason. 
    random_baseline_means = np.ones(len(margins)) * 0.5
    random_baseline_stds = []

    for margin in margins:
        if mode == '<=':
            subset = stereoisomers_df_margins[
                np.round(stereoisomers_df_margins.difference, 1) <= np.round(margin, 1)]  # change to  ==, >=, <=
        elif mode == '>=':
            subset = stereoisomers_df_margins[
                np.round(stereoisomers_df_margins.difference, 1) >= np.round(margin, 1)]  # change to  ==, >=, <=
        elif mode == '==':
            subset = stereoisomers_df_margins[
                np.round(stereoisomers_df_margins.difference, 1) == np.round(margin, 1)]  # change to  ==, >=, <=

        def _match(x):
            pred_scores = np.array(x.mean_predicted_score)
            if np.abs(pred_scores[0] - pred_scores[1]) < difference_threshold:
                return False
            return np.argmin(np.array(x.targets)) == np.argmin(pred_scores)

        top_1 = subset.groupby('SMILES_nostereo').apply(_match)
        random_baseline_std = np.sqrt(
            len(top_1) * 0.5 * 0.5)  # sqrt(npq) -- std of number of guesses expected to be right, when guessing randomly
        random_baseline_stds.append(random_baseline_std / len(top_1))
        acc = sum(top_1 / len(top_1))
        top_1_margins.append(acc)

    random_baseline_stds = np.array(random_baseline_stds)

    return margins, np.array(top_1_margins), random_baseline_means, random_baseline_stds


def mask_chiral_default(data: torch_geometric.data.Data) -> torch_geometric.data.Data:
    """
    Adapted from ChIRo.model.dataset_samplers. It simply masks chiral tags
    (see ChIRo.model.embedding_functions.getNodeFeatures)
    """

    data.x[:, -9:] = 0.0
    data.edge_attr[:, -7:] = 0.0
    return data


def mask_chiral_edge_graph(data: torch_geometric.data.Data) -> torch_geometric.data.Data:
    """
    Mask chiral tags for an edge graph obtained with `to_edge_graph` function. The last 9 values in node embedding
    are chiral tags, and so are the last 7 values in edge embedding (see 'get_chiro_data_from_mol'). This function
    is to be used when `double_embedding` flag is set to true in `to_edge_graph`, so x_{i, j} = x'_i | e'_{i, j} | x'_j.
    """
    edge_attr_dim = 14
    node_attr_dim = 52
    # Masking chiral tags from x'_i in x'_i | e'_{i, j} | x'_j:
    data.x[:, node_attr_dim - 9: node_attr_dim] = 0.0
    # Masking chiral tags from e'_{i, j} in x'_i | e'_{i, j} | x'_j:
    data.x[:, node_attr_dim + edge_attr_dim - 7: node_attr_dim + edge_attr_dim] = 0.0
    # Masking chiral tags from x'_j in x'_i | e'_{i, j} | x'_j:
    data.x[:, node_attr_dim + edge_attr_dim + node_attr_dim - 9: node_attr_dim + edge_attr_dim + node_attr_dim] = 0.0

    # Masking chiral tags from e'_(i, j) in e'_(i, j) | x'_j | e'_{k, j}:
    data.edge_attr[:, edge_attr_dim - 7: edge_attr_dim] = 0.0
    # Masking chiral tags from x'_j in e'_(i, j) | x'_j | e'_{k, j}:
    data.edge_attr[:, edge_attr_dim + node_attr_dim - 9: edge_attr_dim + node_attr_dim] = 0.0
    # Masking chiral tags from e'_{k, j} in e'_(i, j) | x'_j | e'_{k, j}:
    data.edge_attr[:,
    edge_attr_dim + node_attr_dim + edge_attr_dim - 7: edge_attr_dim + node_attr_dim + edge_attr_dim] = 0.0
    return data


def get_number_of_chiral_centers(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol).split('.')[0]
    mol = Chem.MolFromSmiles(smiles)
    return len(Chem.FindMolChiralCenters(mol))


def add_parity_atoms(data: torch_geometric.data.Data) -> torch_geometric.data.Data:
    # Parity atoms are required by Tetra-DMPNN model. It can anly be used with original non-edge graph.
    # Parity atom should map from atom index to CW (+1), CCW (-1) or undefined tetra (0).
    # `data.x[:, -9:]` are one hots encoidngs of chiral tags. `data.x[:, -4]` are CW labels while `data.x[:, -4]`
    # are CCW labels.
    data.parity_atoms = data.x[:, -4] - data.x[:, -3]
    return data


PRE_TRANSFORM_MAPPING = {
    'edge_graph': to_edge_graph,
    'add_parity_atoms': add_parity_atoms
}

CHIRAL_MASKING_MAPPING = {
    'default': mask_chiral_default,
    'edge_graph': mask_chiral_edge_graph,
    'add_parity_atoms': mask_chiral_default
}


class Molecule3DEmbedder:
    """
    Creates rdkit.Mol from a smiles and embeds it in 3D space. Performs some cleaning as data from Tox21 are rather toxic.
    """

    def __init__(self, max_number_of_atoms: int, max_number_of_attempts: int = 5000):
        """
        Args:
            max_number_of_atoms: maximal number of atoms in a molecule. Molecules with more atoms will be omitted.
            max_number_of_attempts: maximal number of attempts during the embedding.
        """
        self.max_number_of_atoms = max_number_of_atoms
        self.max_number_of_attempts = max_number_of_attempts

    def embed(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Embeds the molecule in 3D space.
        Args:
            smiles: a smile representing molecule

        Returns:
            Embedded molecule.
        """

        # Back and forth conversion canonizes the SMILES. After the canonization, the biggest molecule
        # is at the beginning.
        return smiles_to_3d_mol(smiles,
                                max_number_of_atoms=self.max_number_of_atoms,
                                max_number_of_attempts=self.max_number_of_attempts)

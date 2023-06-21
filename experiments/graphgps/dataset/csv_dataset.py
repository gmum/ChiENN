import logging
import os.path as osp

import pandas as pd
import torch
from chainer_chemistry.dataset.splitters.random_splitter import RandomSplitter
from chainer_chemistry.dataset.splitters.scaffold_splitter import ScaffoldSplitter
from torch_geometric.graphgym import cfg
from tqdm import tqdm

from graphgps.dataset.chiral_dataset_base import ChiralDatasetBase
from graphgps.dataset.utils import get_chiro_data_from_mol, Molecule3DEmbedder, download_url_to_path, \
    convert_target_for_task


class CSVDataset(ChiralDatasetBase):
    r"""
    Dataset for tasks defined in csv files. Inspired by `ZINC` dataset from torch_geometric.graphgym.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        name: task name to lookup csv dict.
        mask_chiral_tags: a flag indicating whether to mask chiral tag
        pre_transform_name (str): A name of function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        max_number_of_atoms: maximal number of atoms in a molecule. Molecules with more atoms will be omitted.
    """

    url_dict = {
        'BACE': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
            'raw_name': 'bace.csv',
            'data_column': 'mol',
            'split_type': 'scaffold',
            'target_column': 'Class'
        },
        'Tox21': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
            'raw_name': 'tox21.csv.gz',
            'data_column': 'smiles',
            'split_type': 'random',
            'target_column': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        }
    }

    def __init__(self, root, name, mask_chiral_tags, split='train', pre_transform_name=None, max_number_of_atoms=100):
        self.name = name
        super().__init__(root, mask_chiral_tags, split, pre_transform_name, max_number_of_atoms)

    @property
    def raw_file_names(self):
        return [self.url_dict[self.name]['raw_name']]

    @property
    def processed_dir(self):
        pre_transform = self.pre_transform_name if self.pre_transform_name else ''
        return osp.join(self.root, pre_transform, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        path = osp.join(self.raw_dir, self.raw_file_names[0])
        download_url_to_path(self.url_dict[self.name]['url'], path)

    def split(self, df: pd.DataFrame, data_column: str, split_type: str) -> str:
        if split_type == 'random':
            splitter = RandomSplitter()
        elif split_type == 'scaffold':
            splitter = ScaffoldSplitter()
        else:
            raise NotImplemented(
                f'Split type {split_type} is not allowed. Only random and scaffold splitting are supported!')
        train_idx, valid_idx, test_idx = splitter.train_valid_test_split(df, smiles_list=df[data_column],
                                                                         frac_train=0.7,
                                                                         frac_valid=0.1, frac_test=0.2,
                                                                         seed=0, include_chirality=True,
                                                                         return_index=True)
        df['split'] = None
        df.loc[train_idx, 'split'] = 'Train'
        df.loc[valid_idx, 'split'] = 'Valid'
        df.loc[test_idx, 'split'] = 'Test'
        return 'split'

    def process(self):
        """
        Processes and saves datapoints from the entire dataset.
        """

        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]))
        metadata = self.url_dict[self.name]
        data_column = metadata['data_column']
        target_column = metadata['target_column']
        split_type = metadata['split_type']
        if split_type == 'predefined':
            split_column = metadata['split_column']
        else:
            split_column = self.split(df, data_column, split_type)

        embedder = Molecule3DEmbedder(self.max_number_of_atoms)
        for split in ['Train', 'Valid', 'Test']:
            split_df = df[df[split_column] == split]
            split = split.lower()
            split = 'val' if split == 'valid' else split
            omitted = 0
            data_list = []
            for index, row in tqdm(split_df.iterrows(), desc=f'Processing {split} dataset', total=len(split_df)):
                smiles = row[data_column]
                mol = embedder.embed(smiles)
                if mol is None:
                    omitted += 1
                    continue

                try:
                    data = get_chiro_data_from_mol(mol)
                except Exception as e:
                    omitted += 1
                    logging.warning(f'Omitting molecule {smiles} as cannot be properly embedded. '
                                    f'The original error message was: {e}.')  # probably it does not have sufficient number of paths of length 3.
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data.y = torch.tensor(row[target_column])
                data.y = convert_target_for_task(data.y, cfg.dataset.task_type)
                data_list.append(data)
            logging.warning(f'Total omitted molecules for {split}: {omitted}.')
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))


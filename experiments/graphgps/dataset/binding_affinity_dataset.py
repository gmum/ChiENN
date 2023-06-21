import logging
import os.path as osp
import pickle

import pandas as pd
import torch
from tqdm import tqdm

from graphgps.dataset.chiral_dataset_base import ChiralDatasetBase
from graphgps.dataset.utils import download_url_to_path, get_chiro_data_from_mol, Molecule3DEmbedder


class BindingAffinity(ChiralDatasetBase):
    r"""
    Dataset for chiral binding affinity classification task from ChIRo paper.
    Inspired by `ZINC` dataset from torch_geometric.graphgym.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        single_conformer: a flag indicating whether to group conformers by enantiomer id and use only one conformer.
            Should be set to True, when model is conformer invariant to save some RAM.
        mask_chiral_tags: a flag indicating whether to mask chiral tag
        pre_transform_name (str): A name of function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    urls_dict = {
        'train': 'https://figshare.com/ndownloader/files/30975697?private_link=e23be65a884ce7fc8543',
        'val': 'https://figshare.com/ndownloader/files/30975706?private_link=e23be65a884ce7fc8543',
        'test': 'https://figshare.com/ndownloader/files/30975682?private_link=e23be65a884ce7fc8543'
    }

    def __init__(self, root, single_conformer, mask_chiral_tags, split='train', single_enantiomer=False,
                 pre_transform_name=None, max_number_of_atoms=100):
        self.single_conformer = single_conformer
        self.single_enantiomer = single_enantiomer
        super().__init__(root, mask_chiral_tags, split, pre_transform_name, max_number_of_atoms)
        self.dataframe = pd.read_csv(osp.join(self.processed_dir, f'{split}.csv'))

    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']

    @property
    def processed_dir(self):
        name = 'single_conformer' if self.single_conformer else 'all_conformers'
        name = f'{name}+single_enantiomer' if self.single_enantiomer else name
        pre_transform = self.pre_transform_name if self.pre_transform_name else ''
        return osp.join(self.root, name, pre_transform, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt', 'train.csv', 'val.csv', 'test.csv']

    def download(self):
        for split, url in self.urls_dict.items():
            split_pickle_path = osp.join(self.raw_dir, f'{split}.pickle')
            download_url_to_path(url, split_pickle_path)

    def process(self):
        """
        Processes and saves datapoints from the entire dataset. It additionally saves original dataframes from
        downloaded pickles which are then used in `SingleConformerBatchSampler` in `get_custom_loader`.
        """
        embedder = Molecule3DEmbedder(self.max_number_of_atoms)
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                split_df = pickle.load(f)

            if self.single_conformer:
                split_df = split_df.drop_duplicates(subset='ID')

            if self.single_enantiomer:
                split_df = split_df.groupby('SMILES_nostereo').sample(n=1, random_state=0)

            data_list = []
            omitted = 0
            to_remove = set()
            for index, row in tqdm(split_df.iterrows(), desc=f'Processing {split} dataset', total=len(split_df)):
                smiles_nonstereo = row['SMILES_nostereo']
                if smiles_nonstereo in to_remove:
                    omitted += 1
                    continue

                smiles = row['ID']
                mol = embedder.embed(smiles)
                if mol is None:
                    omitted += 1
                    to_remove.add(smiles_nonstereo)
                    continue

                try:
                    data = get_chiro_data_from_mol(mol)
                except Exception as e:
                    omitted += 1
                    to_remove.add(smiles_nonstereo)
                    logging.warning(f'Omitting molecule {smiles} as cannot be properly embedded. '
                                    f'The original error message was: {e}.')  # probably it does not have sufficient number of paths of length 3.
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data.y = torch.tensor(row['top_score']).float()
                data_list.append((smiles_nonstereo, data))
            logging.warning(f'Total omitted molecules for {split}: {omitted}.')
            data_list = [data for smiles, data in data_list if smiles not in to_remove]
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            split_df = split_df.drop(columns='rdkit_mol_cistrans_stereo')
            split_df = split_df[~split_df['SMILES_nostereo'].isin(to_remove)]
            split_df.to_csv(osp.join(self.processed_dir, f'{split}.csv'), index=None)


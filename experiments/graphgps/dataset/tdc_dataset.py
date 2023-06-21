import logging
import os.path as osp

import tdc.single_pred
import torch
from tdc.utils import create_scaffold_split
from torch_geometric.graphgym import cfg
from tqdm import tqdm

from graphgps.dataset.chiral_dataset_base import ChiralDatasetBase
from graphgps.dataset.utils import get_chiro_data_from_mol, Molecule3DEmbedder, convert_target_for_task
from graphgps.dataset.utils import get_number_of_chiral_centers


class TDC(ChiralDatasetBase):
    r"""
    Dataset for tasks from Therapeutics Data Commons (TDC) (https://tdcommons.ai/).
    See https://tdcommons.ai/single_pred_tasks/tox/#tox21 for use case of TDC.
    Inspired by `ZINC` dataset from torch_geometric.graphgym.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        tdc_type: task type used to get appropriate constructor. In practice, it will be "Tox" or "ADME".
        tdc_dataset_name: dataset name in TDC framework.
        tdc_assay_name: assay/label name. Used in datasets with many assays/labels.
        mask_chiral_tags: a flag indicating whether to mask chiral tag
        pre_transform_name (str): A name of function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        max_number_of_atoms: maximal number of atoms in a molecule. Molecules with more atoms will be omitted.
    """

    def __init__(self, root, tdc_type, tdc_dataset_name, tdc_assay_name, mask_chiral_tags, split='train',
                 pre_transform_name=None, max_number_of_atoms=100, min_number_of_chiral_centers=0):
        self.tdc_type = tdc_type
        self.tdc_dataset_name = tdc_dataset_name
        self.tdc_assay_name = tdc_assay_name
        self.min_number_of_chiral_centers = min_number_of_chiral_centers
        super().__init__(root, mask_chiral_tags, split, pre_transform_name, max_number_of_atoms)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        pre_transform = self.pre_transform_name if self.pre_transform_name else ''
        n = self.min_number_of_chiral_centers
        chirality = f'at_least_{n}_chiral_centers' if n > 0 else ''
        return osp.join(self.root, pre_transform, chirality, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        """
        Processes and saves datapoints from the entire dataset.
        """

        task = getattr(tdc.single_pred, self.tdc_type)
        data = task(name=self.tdc_dataset_name, label_name=self.tdc_assay_name if self.tdc_assay_name else None)
        if self.min_number_of_chiral_centers > 0:
            df = data.get_data()
            df['n_chirals'] = df['Drug'].apply(get_number_of_chiral_centers)
            df = df[df['n_chirals'] >= self.min_number_of_chiral_centers]
            split_dict = create_scaffold_split(df, seed=0, frac=[0.7, 0.1, 0.2], entity=data.entity1_name)
        else:
            split_dict = data.get_split(method='scaffold', seed=0, frac=[0.7, 0.1, 0.2])
        embedder = Molecule3DEmbedder(self.max_number_of_atoms)
        for split, split_df in split_dict.items():
            split = 'val' if split == 'valid' else split
            omitted = 0
            data_list = []
            for index, row in tqdm(split_df.iterrows(), desc=f'Processing {split} dataset', total=len(split_df)):
                smiles = row.Drug
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

                data.y = torch.tensor(row.Y)
                data.y = convert_target_for_task(data.y, cfg.dataset.task_type, cfg.dataset.scale_label)

                data_list.append(data)
            logging.warning(f'Total omitted molecules for {split}: {omitted}.')
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))


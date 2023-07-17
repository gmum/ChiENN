import logging
import os.path as osp
import zipfile

import pandas as pd
import torch
from torch_geometric.graphgym import cfg
from tqdm import tqdm

from graphgps.dataset.chiral_dataset_base import ChiralDatasetBase
from graphgps.dataset.utils import (
    convert_target_for_task,
    download_url_to_path,
    get_chiro_data_from_mol, Molecule3DEmbedder,
)


class OGB(ChiralDatasetBase):
    r"""
    Dataset for tasks from Open Graph Benchmark (OGB) (https://ogb.stanford.edu/).
    Inspired by `ZINC` dataset from torch_geometric.graphgym.

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        ogb_dataset_name: dataset name in OGB framework.
        mask_chiral_tags: a flag indicating whether to mask chiral tag
        pre_transform_name (str): A name of function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        max_number_of_atoms: maximal number of atoms in a molecule. Molecules with more atoms will be omitted.
    """

    url_dict = {
        "hiv": "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip",
        "pcba": "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/pcba.zip",
    }

    def __init__(
        self,
        root,
        ogb_dataset_name,
        mask_chiral_tags,
        split="train",
        pre_transform_name=None,
        max_number_of_atoms=100,
    ):
        if ogb_dataset_name not in self.url_dict.keys():
            raise NotImplementedError(
                "Expected ogb_dataset_name in "
                f"{set(self.url_dict.keys())}"
                ", got "
                f"'{ogb_dataset_name}'."
            )

        self.ogb_dataset_name = ogb_dataset_name

        super().__init__(
            root=root,
            mask_chiral_tags=mask_chiral_tags,
            split=split,
            pre_transform_name=pre_transform_name,
            max_number_of_atoms=max_number_of_atoms,
        )

    @property
    def raw_file_names(self):
        return [
            self.url_dict[self.ogb_dataset_name].split("/")[-1],
            self.url_dict[self.ogb_dataset_name].split("/")[-1].replace(".zip", ""),
        ]

    @property
    def processed_dir(self):
        pre_transform = self.pre_transform_name if self.pre_transform_name else ""
        return osp.join(self.root, pre_transform, "processed")

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def download(self):
        zip_path = osp.join(self.raw_dir, self.raw_file_names[0])
        ext_path = osp.join(self.raw_dir, self.raw_file_names[1])

        download_url_to_path(self.url_dict[self.ogb_dataset_name], zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(ext_path)

    def process(self):
        """
        Processes and saves datapoints from the entire dataset.
        """

        dataset_dir = osp.join(
            self.raw_dir, self.raw_file_names[1], self.ogb_dataset_name
        )
        data_path = osp.join(dataset_dir, "mapping", "mol.csv.gz")
        split_path = osp.join(dataset_dir, "split", "scaffold")

        df = pd.read_csv(data_path).iloc[:, :-1]
        split_dict = {}

        for split in ["train", "valid", "test"]:
            idx_df = pd.read_csv(osp.join(split_path, f"{split}.csv.gz"), header=None)
            split_dict[split] = df.iloc[idx_df[0]]

        embedder = Molecule3DEmbedder(self.max_number_of_atoms)
        for split, split_df in split_dict.items():
            split = "val" if split == "valid" else split
            omitted = 0
            data_list = []
            for index, row in tqdm(
                split_df.iterrows(),
                desc=f"Processing {split} dataset",
                total=len(split_df),
            ):
                smiles = row.smiles
                mol = embedder.embed(smiles)
                if mol is None:
                    omitted += 1
                    continue

                try:
                    data = get_chiro_data_from_mol(mol)
                except Exception as e:
                    omitted += 1
                    logging.warning(
                        f"Omitting molecule {smiles} as cannot be properly embedded. "
                        f"The original error message was: {e}."
                    )  # probably it does not have sufficient number of paths of length 3.
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data.y = torch.tensor(row.iloc[:-1])
                data.y = convert_target_for_task(data.y, cfg.dataset.task_type)

                data_list.append(data)
            logging.warning(f"Total omitted molecules for {split}: {omitted}.")
            torch.save(
                self.collate(data_list), osp.join(self.processed_dir, f"{split}.pt")
            )

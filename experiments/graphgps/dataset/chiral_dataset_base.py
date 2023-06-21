import abc
import copy
import os.path as osp

import torch
import torch_geometric.data
from torch_geometric.data import InMemoryDataset

from graphgps.dataset.utils import PRE_TRANSFORM_MAPPING, CHIRAL_MASKING_MAPPING


class ChiralDatasetBase(InMemoryDataset, abc.ABC):
    r"""
    Dataset base class that
    """

    def __init__(
        self,
        root,
        mask_chiral_tags,
        split="train",
        pre_transform_name=None,
        max_number_of_atoms=100,
    ):
        assert split in ["train", "val", "test"]
        self.mask_chiral_tags = mask_chiral_tags
        self.pre_transform_name = (
            pre_transform_name if pre_transform_name else "default"
        )
        pre_transform = PRE_TRANSFORM_MAPPING.get(self.pre_transform_name)
        self.mask_chiral_fn = CHIRAL_MASKING_MAPPING.get(self.pre_transform_name)
        self.max_number_of_atoms = max_number_of_atoms
        super().__init__(
            root, transform=None, pre_transform=pre_transform, pre_filter=None
        )
        self.data, self.slices = torch.load(osp.join(self.processed_dir, f"{split}.pt"))

    def __getitem__(self, idx: int):
        """
        Standard getitem with chiral tags masking.
        """

        # for some reason it returns something different than data.Data at the very beginning of the training:
        data = super().__getitem__(idx)
        if isinstance(data, torch_geometric.data.Data):
            if self.mask_chiral_tags:
                data = self.mask_chiral_fn(data)
        return data

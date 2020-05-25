"""Multi MNIST dataset."""
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch

from ssd.data.datasets.base import BaseDataset, DataTransformType, TargetTransformType


class MultiScaleMNIST(BaseDataset):
    """Multi-scale MNIST dataset. Requires the """

    def __init__(
        self,
        data_dir: str,
        data_transform: DataTransformType = None,
        target_transform: TargetTransformType = None,
        subset: str = "train",
        h5_filename: str = "multiscalemnist.h5",
    ):
        super().__init__(
            str(Path(data_dir).joinpath("mnist")),
            data_transform,
            target_transform,
            subset,
        )
        self.dataset_file = self.data_dir.joinpath(h5_filename)
        with h5py.File(self.dataset_file, "r") as file:
            self.dataset_length = len(file[self.subset]["images"])
        self.dataset: Optional[h5py.File] = None

    def __len__(self):
        """Get dataset length."""
        return self.dataset_length

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_file, "r")[self.subset]
        return super().__getitem__(item)

    def _get_image(self, item: int) -> torch.Tensor:
        image = self.dataset["images"][item] / 255  # type: ignore
        return torch.from_numpy(image).unsqueeze(0).float()

    def _get_annotation(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes = self.dataset["boxes"][item]  # type: ignore
        labels = self.dataset["labels"][item]  # type: ignore
        mask = np.where(labels != -1)
        return (
            torch.from_numpy(boxes[mask]).float(),
            torch.from_numpy(labels[mask] + 1).long(),  # 0 must be background class
        )

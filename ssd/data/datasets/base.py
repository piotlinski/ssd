"""Base class for dataset."""
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.utils.data as data

DataTransformType = Optional[
    Callable[
        [torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    ]
]
TargetTransformType = Optional[
    Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
]


class BaseDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        data_transform: DataTransformType = None,
        target_transform: TargetTransformType = None,
        subset: str = "train",
    ):
        """
        :param data_dir: directory with data for the dataset
        :param data_transform: transforms to apply to both images and targets
        :param target_transform: transforms to apply to targets
        :param subset: subset to use
        """
        self.data_dir: Path = Path(data_dir)
        self.data_transform: DataTransformType = data_transform
        self.target_transform: TargetTransformType = target_transform
        self.subset = subset

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self._get_image(item)
        boxes, labels = self._get_annotation(item)
        if self.data_transform is not None:
            image, boxes, labels = self.data_transform(image, boxes, labels)
        if self.target_transform is not None:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def _get_image(self, item: int) -> torch.Tensor:
        """Get image from the dataset."""
        raise NotImplementedError

    def _get_annotation(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get annotations (boxes and labels) from the dataset."""
        raise NotImplementedError
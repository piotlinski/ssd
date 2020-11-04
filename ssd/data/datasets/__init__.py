from typing import Dict, Type

from ssd.data.datasets.base import BaseDataset, onehot_labels
from ssd.data.datasets.coco import COCODetection
from ssd.data.datasets.mnist import MultiScaleMNIST

datasets: Dict[str, Type[BaseDataset]] = {
    "MultiscaleMNIST": MultiScaleMNIST,
    "COCO": COCODetection,
}

__all__ = ["onehot_labels", "datasets"]

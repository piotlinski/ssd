from typing import Dict, Type

from pytorch_ssd.data.datasets.base import BaseDataset, onehot_labels
from pytorch_ssd.data.datasets.clevr import CLEVR
from pytorch_ssd.data.datasets.coco import COCODetection
from pytorch_ssd.data.datasets.mnist import MultiScaleMNIST

datasets: Dict[str, Type[BaseDataset]] = {
    "MNIST": MultiScaleMNIST,
    "COCO": COCODetection,
    "CLEVR": CLEVR,
}

__all__ = ["onehot_labels", "datasets"]

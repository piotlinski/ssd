from typing import Dict, Type

from pyssd.data.datasets.base import BaseDataset, onehot_labels
from pyssd.data.datasets.clevr import CLEVR
from pyssd.data.datasets.coco import COCODetection
from pyssd.data.datasets.mnist import MultiScaleMNIST

datasets: Dict[str, Type[BaseDataset]] = {
    "MultiscaleMNIST": MultiScaleMNIST,
    "COCO": COCODetection,
    "CLEVR": CLEVR,
}

__all__ = ["onehot_labels", "datasets"]

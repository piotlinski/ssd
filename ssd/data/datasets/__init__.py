from typing import Dict, Type

from ssd.data.datasets.base import BaseDataset
from ssd.data.datasets.mnist import MultiScaleMNIST

datasets: Dict[str, Type[BaseDataset]] = {"MultiscaleMNIST": MultiScaleMNIST}

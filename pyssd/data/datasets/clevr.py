"""CLEVR dataset for detection."""
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import PIL
import torch

from pyssd.data.datasets.base import BaseDataset, DataTransformType, TargetTransformType


class CLEVR(BaseDataset):
    """CLEVR dataset."""

    CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"

    datasets = {
        "train": ("images/train", "bboxes/train.json"),
        "test": ("images/val", "bboxes/val.json"),
    }

    CLASS_LABELS = [
        "large gray rubber cube",
        "large gray rubber sphere",
        "large gray rubber cylinder",
        "large gray metal cube",
        "large gray metal sphere",
        "large gray metal cylinder",
        "large red rubber cube",
        "large red rubber sphere",
        "large red rubber cylinder",
        "large red metal cube",
        "large red metal sphere",
        "large red metal cylinder",
        "large blue rubber cube",
        "large blue rubber sphere",
        "large blue rubber cylinder",
        "large blue metal cube",
        "large blue metal sphere",
        "large blue metal cylinder",
        "large green rubber cube",
        "large green rubber sphere",
        "large green rubber cylinder",
        "large green metal cube",
        "large green metal sphere",
        "large green metal cylinder",
        "large brown rubber cube",
        "large brown rubber sphere",
        "large brown rubber cylinder",
        "large brown metal cube",
        "large brown metal sphere",
        "large brown metal cylinder",
        "large purple rubber cube",
        "large purple rubber sphere",
        "large purple rubber cylinder",
        "large purple metal cube",
        "large purple metal sphere",
        "large purple metal cylinder",
        "large cyan rubber cube",
        "large cyan rubber sphere",
        "large cyan rubber cylinder",
        "large cyan metal cube",
        "large cyan metal sphere",
        "large cyan metal cylinder",
        "large yellow rubber cube",
        "large yellow rubber sphere",
        "large yellow rubber cylinder",
        "large yellow metal cube",
        "large yellow metal sphere",
        "large yellow metal cylinder",
        "small gray rubber cube",
        "small gray rubber sphere",
        "small gray rubber cylinder",
        "small gray metal cube",
        "small gray metal sphere",
        "small gray metal cylinder",
        "small red rubber cube",
        "small red rubber sphere",
        "small red rubber cylinder",
        "small red metal cube",
        "small red metal sphere",
        "small red metal cylinder",
        "small blue rubber cube",
        "small blue rubber sphere",
        "small blue rubber cylinder",
        "small blue metal cube",
        "small blue metal sphere",
        "small blue metal cylinder",
        "small green rubber cube",
        "small green rubber sphere",
        "small green rubber cylinder",
        "small green metal cube",
        "small green metal sphere",
        "small green metal cylinder",
        "small brown rubber cube",
        "small brown rubber sphere",
        "small brown rubber cylinder",
        "small brown metal cube",
        "small brown metal sphere",
        "small brown metal cylinder",
        "small purple rubber cube",
        "small purple rubber sphere",
        "small purple rubber cylinder",
        "small purple metal cube",
        "small purple metal sphere",
        "small purple metal cylinder",
        "small cyan rubber cube",
        "small cyan rubber sphere",
        "small cyan rubber cylinder",
        "small cyan metal cube",
        "small cyan metal sphere",
        "small cyan metal cylinder",
        "small yellow rubber cube",
        "small yellow rubber sphere",
        "small yellow rubber cylinder",
        "small yellow metal cube",
        "small yellow metal sphere",
        "small yellow metal cylinder",
    ]
    OBJECT_LABEL = "object"

    def __init__(
        self,
        data_dir: str,
        data_transform: DataTransformType = None,
        target_transform: TargetTransformType = None,
        subset: str = "train",
    ):
        super().__init__(data_dir, data_transform, target_transform, subset)
        self.image_dir, annotations_file = self.datasets[subset]
        with self.data_dir.joinpath(annotations_file).open("r") as fp:
            self.annotations = json.load(fp)
        self.image_names = list(self.annotations.keys())

    def __len__(self):
        """Get dataset length."""
        return len(self.image_names)

    def _get_image(self, item: int) -> torch.Tensor:
        image_file = self.image_names[item]
        image_path = self.data_dir.joinpath(self.image_dir).joinpath(image_file)
        image = np.array(PIL.Image.open(image_path).convert("RGB")) / 255
        return torch.tensor(image)

    def _get_annotation(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self.image_names[item]
        ann = self.annotations(image_file)
        boxes = torch.tensor(
            list(zip(ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"])),
            dtype=torch.float32,
        )
        labels = torch.tensor(ann["class"], dtype=torch.int64)
        return boxes, labels

    @classmethod
    def download(cls, path: str):
        """Download and extract CLEVR dataset."""
        data_path = Path(path)
        data_path.mkdir(exist_ok=True)
        filename = cls.CLEVR_URL.split("/")[-1]
        target_path = data_path.joinpath(filename)
        cmd = f"curl {cls.CLEVR_URL} -o {str(target_path)}"
        subprocess.call(cmd, shell=True)
        with zipfile.ZipFile(str(target_path)) as zf:
            zf.extractall(path=data_path)

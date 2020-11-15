"""CLEVR dataset for detection."""
import json
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import PIL
import torch

from pytorch_ssd.data.datasets.base import (
    BaseDataset,
    DataTransformType,
    TargetTransformType,
)


class CLEVR(BaseDataset):
    """CLEVR dataset."""

    sizes = ["large", "small"]
    colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    materials = ["rubber", "metal"]
    shapes = ["cube", "sphere", "cylinder"]

    CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"

    datasets = {
        "train": ("images/train", "scenes/CLEVR_train_scenes.json"),
        "test": ("images/val", "scenes/CLEVR_val_scenes.json"),
    }

    CLASS_LABELS: List[str] = []
    OBJECT_LABEL = "object"

    def __init__(
        self,
        data_dir: str,
        data_transform: DataTransformType = None,
        target_transform: TargetTransformType = None,
        subset: str = "train",
    ):
        super().__init__(data_dir, data_transform, target_transform, subset)
        if not self.CLASS_LABELS:
            self.CLASS_LABELS = [
                f"{size} {color} {material} {shape}"
                for size in self.sizes
                for color in self.colors
                for material in self.materials
                for shape in self.shapes
            ]
        self.image_dir, annotations_file = self.datasets[subset]
        with self.data_dir.joinpath(annotations_file).open("r") as fp:
            self.annotations = json.load(fp)["scenes"]
        self.image_names = [ann["image_filename"] for ann in self.annotations]

    def __len__(self):
        """Get dataset length."""
        return len(self.image_names)

    def _get_image(self, item: int) -> torch.Tensor:
        image_file = self.image_names[item]
        image_path = self.data_dir.joinpath(self.image_dir).joinpath(image_file)
        image = np.array(PIL.Image.open(image_path).convert("RGB")) / 255
        return torch.tensor(image)

    def extract_bbox_and_label(
        self, scene: Dict[str, Any]
    ) -> Dict[str, List[Union[int, float]]]:
        """Create bbox and label from scene annotation.

        .. note: sourced from
           https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py

        :param scene: scene annotation dict according to CLEVR format
        :return: Dict with list of bbox params: x_min, y_min, x_max, y_max, class
        """
        annotation: Dict[str, List[Union[int, float]]] = defaultdict(list)
        objs = scene["objects"]
        rotation = scene["directions"]["right"]

        for obj in objs:
            x, y, _ = obj["pixel_coords"]
            x1, y1, z1 = obj["3d_coords"]

            cos_theta, sin_theta, _ = rotation

            x1 = x1 * cos_theta + y1 * sin_theta
            y1 = x1 * -sin_theta + y1 * cos_theta

            height_d = height_u = width_l = width_r = 6.9 * z1 * (15 - y1) / 2.0

            if obj["shape"] == "cylinder":
                d = 9.4 + y1
                h = 6.4
                s = z1

                height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
                height_d = height_u * (h - s + d) / (h + s + d)

                width_l *= 11 / (10 + y1)
                width_r = width_l

            elif obj["shape"] == "cube":
                height_u *= 1.3 * 10 / (10 + y1)
                height_d = height_u
                width_l = height_u
                width_r = height_u

            class_name = (
                f"{obj['size']} {obj['color']} {obj['material']} {obj['shape']}"
            )
            annotation["class"].append(self.CLASS_LABELS.index(class_name) + 1)
            annotation["x_min"].append(max(0, x - width_l))
            annotation["y_min"].append(max(0, y - height_d))
            annotation["x_max"].append(min(480, x + width_r))
            annotation["y_max"].append(min(320, y + height_u))

        return annotation

    def _get_annotation(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scene = self.annotations[item]
        ann = self.extract_bbox_and_label(scene)
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

"""COCO dataset."""
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import PIL
import torch
from pycocotools.coco import COCO

from pytorch_ssd.data.datasets.base import (
    BaseDataset,
    DataTransformType,
    TargetTransformType,
)


class COCODetection(BaseDataset):
    """Multi-scale MNIST dataset."""

    COCO_URLS = {
        ("images/", "http://images.cocodataset.org/zips/train2017.zip"),
        ("images/", "http://images.cocodataset.org/zips/val2017.zip"),
        ("/", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
    }
    datasets = {
        "train": (
            "images/train2017",
            "annotations/instances_train2017.json",
        ),
        "test": (
            "images/val2017",
            "annotations/instances_val2017.json",
        ),
    }
    CLASS_LABELS = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
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
        self.image_dir, self.annotations_file = self.datasets[subset]
        try:
            sys.stdout = open(os.devnull, "w")
            self.coco = COCO(str(self.data_dir.joinpath(self.annotations_file)))
        finally:
            sys.stdout = sys.__stdout__
        self.category_idx_mapping = {
            coco_idx: idx for idx, coco_idx in enumerate(self.coco.getCatIds(), start=1)
        }
        self.indices = list(self.coco.imgToAnns.keys())

    def __len__(self):
        """Get dataset length."""
        return len(self.coco.imgToAnns)

    def _get_image(self, item: int) -> torch.Tensor:
        index = self.indices[item]
        image_file = self.coco.loadImgs(index)[0]["file_name"]
        image_path = self.data_dir.joinpath(self.image_dir).joinpath(image_file)
        image = np.array(PIL.Image.open(image_path).convert("RGB")) / 255
        return torch.tensor(image)

    @staticmethod
    def coco_bbox_to_corner_bbox(bbox: List[float]) -> List[float]:
        """Convert coco bbox to X1Y1X2Y2 form."""
        x1, y1, w, h = bbox
        return [x1, y1, x1 + w, y1 + h]

    def _get_annotation(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=index))
        annotations = [obj for obj in annotations if obj["iscrowd"] == 0]
        boxes = torch.tensor(
            [self.coco_bbox_to_corner_bbox(obj["bbox"]) for obj in annotations],
            dtype=torch.float32,
        ).view((-1, 4))
        labels = torch.tensor(
            [self.category_idx_mapping[obj["category_id"]] for obj in annotations],
            dtype=torch.int64,
        ).view((-1,))
        keep_inds = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        return boxes[keep_inds], labels[keep_inds]

    @classmethod
    def download(cls, path: str):
        """Download and extract COCO dataset """
        data_path = Path(path)
        data_path.mkdir(exist_ok=True)
        for target_dir, url in cls.COCO_URLS:
            filename = url.split("/")[-1]
            target_path = data_path.joinpath(filename)
            cmd = f"curl {url} -o {str(target_path)}"
            subprocess.call(cmd, shell=True)
            with zipfile.ZipFile(str(target_path)) as zf:
                extract_path = data_path.joinpath(target_dir)
                extract_path.mkdir(exist_ok=True)
                zf.extractall(path=extract_path)
            target_path.unlink()

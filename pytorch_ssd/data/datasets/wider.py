"""WIDER FACE dataset."""
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import PIL
import torch
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_file_from_google_drive,
    extract_archive,
)

from pytorch_ssd.data.datasets.base import (
    BaseDataset,
    DataTransformType,
    TargetTransformType,
)


class WIDERFace(BaseDataset):
    """WIDER FACE detection dataset."""

    WIDER_TRAIN = {
        "id": "0B6eKvaijfFUDQUUwd21EckhUbWs",
        "md5": "3fedf70df600953d25982bcd13d91ba2",
        "filename": "WIDER_train.zip",
    }
    WIDER_TEST = {
        "id": "0B6eKvaijfFUDd3dIRmpvSk8tLUk",
        "md5": "dfa7d7e790efa35df3788964cf0bbaea",
        "filename": "WIDER_val.zip",
    }
    WIDER_ANNS = {
        "url": (
            "http://mmlab.ie.cuhk.edu.hk"
            "/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip"
        ),
        "md5": "0e3767bcf0e326556d407bf5bff5d27c",
        "filename": "wider_face_split.zip",
    }

    datasets = {
        "train": ("WIDER_train/images", "wider_face_split/wider_face_train_bbx_gt.txt"),
        "test": ("WIDER_val/images", "wider_face_split/wider_face_val_bbx_gt.txt"),
    }

    CLASS_LABELS = ["face"]
    OBJECT_LABEL = "face"

    def __init__(
        self,
        data_dir: str,
        data_transform: DataTransformType = None,
        target_transform: TargetTransformType = None,
        subset: str = "train",
    ):
        super().__init__(data_dir, data_transform, target_transform, subset)
        self.image_dir, annotations_file = self.datasets[subset]
        self.annotations = self.parse_wider_annotations(
            self.data_dir.joinpath(annotations_file)
        )
        self.image_files = list(self.annotations.keys())

    def __len__(self):
        """Get dataset length."""
        return len(self.annotations)

    def _get_image(self, item: int) -> torch.Tensor:
        image_path = self.image_files[item]
        image = np.array(PIL.Image.open(image_path).convert("RGB")) / 255
        return torch.tensor(image)

    def _get_annotation(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes, labels = self.annotations[self.image_files[item]]
        return boxes.float(), labels.long()

    def parse_wider_annotations(self, annotations_file: Path) -> Dict[Path, Any]:
        """Parse WIDER annotation file for training detection model."""
        annotations = {}
        with annotations_file.open("r") as f:
            file_name_line, num_boxes_line, box_annotation_line = True, False, False
            num_boxes, box_counter = 0, 0
            labels = []
            for line in f.readlines():
                line = line.rstrip()
                if file_name_line:
                    img_path = self.data_dir.joinpath(
                        self.datasets[self.subset][0]
                    ).joinpath(line)
                    file_name_line = False
                    num_boxes_line = True
                elif num_boxes_line:
                    num_boxes = int(line)
                    num_boxes_line = False
                    box_annotation_line = True
                elif box_annotation_line:
                    box_counter += 1
                    line_split = line.split(" ")
                    line_values = [float(x) for x in line_split]
                    labels.append(line_values)
                    if box_counter >= num_boxes:
                        bboxes = []
                        box_annotation_line = False
                        file_name_line = True
                        for label in labels:
                            x0, y0, w, h = label[0:4]
                            if w == 0 or h == 0:
                                continue
                            bboxes.append([x0, y0, x0 + w, y0 + h])
                        if bboxes:
                            annotations[img_path] = (
                                torch.tensor(bboxes),
                                torch.ones(len(bboxes), dtype=torch.int64),
                            )
                        box_counter = 0
                        labels = []
        return annotations

    @classmethod
    def download(cls, path: str):
        """Download and extract WIDER FACE dataset."""
        data_path = Path(path)
        data_path.mkdir(exist_ok=True)
        for dataset in [cls.WIDER_TRAIN, cls.WIDER_TEST]:
            download_file_from_google_drive(
                file_id=dataset["id"],
                root=str(data_path),
                filename=dataset["filename"],
                md5=dataset["md5"],
            )
            downloaded = data_path.joinpath(dataset["filename"])
            extract_archive(downloaded)

        download_and_extract_archive(
            url=cls.WIDER_ANNS["url"],
            download_root=str(data_path),
            filename=cls.WIDER_ANNS["filename"],
            md5=cls.WIDER_ANNS["md5"],
        )

"""MOT15 dataset."""
import subprocess
import zipfile
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import PIL
import torch

from pytorch_ssd.data.datasets.base import (
    BaseDataset,
    DataTransformType,
    TargetTransformType,
)


class MOT15(BaseDataset):
    """MOT15 detection dataset."""

    MOT15_URL = "https://motchallenge.net/data/MOT15.zip"

    CLASS_LABELS = ["person"]
    OBJECT_LABEL = "person"

    def __init__(
        self,
        data_dir: str,
        data_transform: DataTransformType = None,
        target_transform: TargetTransformType = None,
        subset: str = "train",
    ):
        super().__init__(data_dir, data_transform, target_transform, subset)
        self._infos = {}
        self.annotations = self._prepare_annotations()[self.subset]
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        """Get dataset length."""
        return len(self.image_paths)

    def _get_image(self, item: int) -> torch.Tensor:
        image_path = self.image_paths[item]
        image = np.array(PIL.Image.open(image_path).convert("RGB")) / 255
        return torch.tensor(image)

    def _get_annotation(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        annotation = self.annotations[self.image_paths[item]]
        boxes = torch.tensor(annotation, dtype=torch.float32)
        labels = torch.ones(boxes.shape[0], dtype=torch.int64)
        return boxes, labels

    @staticmethod
    def _parse_mot15_line(
        line: str, width: float, height: float
    ) -> Tuple[int, int, float, float, float, float]:
        """Create X1Y1X2Y2 annotation from MOT15 line."""
        frame_id, obj_id, x1, y1, w, h, *_ = line.split(",")
        return (
            int(frame_id),
            int(obj_id),
            max(float(x1), 0),
            max(float(y1), 0),
            min(float(x1) + float(w), width),
            min(float(y1) + float(h), height),
        )

    @staticmethod
    def _parse_ini_file(ini_file: Path) -> Tuple[str, int, int, str]:
        """Parse ini file and return image directory, image width, height and ext."""
        info = ConfigParser()
        with ini_file.open("r") as fp:
            info.read_file(fp)
        parsed = info["Sequence"]
        return (
            parsed["imDir"],
            int(parsed["imWidth"]),
            int(parsed["imHeight"]),
            parsed["imExt"],
        )

    def _parse_mot15_file(self, file: Path) -> Dict[Path, List[List[float]]]:
        """Read annotation from file and return filename and annotations."""
        annotations = defaultdict(list)

        ini_file = file.parents[1] / "seqinfo.ini"
        if ini_file not in self._infos:
            self._infos[ini_file] = self._parse_ini_file(ini_file)

        directory, width, height, extension = self._infos[ini_file]

        with file.open("r") as fp:
            for line in fp:
                frame_id, _, x1, y1, x2, y2 = self._parse_mot15_line(
                    line, width=width, height=height
                )
                filename = file.parents[1] / directory / f"{frame_id:06}{extension}"
                annotations[filename].append([x1, y1, x2, y2])
        return annotations

    def _prepare_annotations(self):
        """Prepare dictionary of all files with their annotations."""
        all_annotations = {"train": {}, "test": {}}
        for annotations_file in self.data_dir.glob("*/*/gt/gt.txt"):
            for index, (key, value) in enumerate(
                self._parse_mot15_file(annotations_file).items()
            ):
                if index % 10 == 0:
                    all_annotations["test"][key] = value
                else:
                    all_annotations["train"][key] = value
        return all_annotations

    @classmethod
    def download(cls, path: str):
        """Download and extract MOT15 dataset."""
        data_path = Path(path)
        data_path.mkdir(exist_ok=True)
        filename = cls.MOT15_URL.split("/")[-1]
        target_path = data_path.joinpath(filename)
        cmd = f"curl {cls.MOT15_URL} -o {str(target_path)}"
        subprocess.call(cmd, shell=True)
        with zipfile.ZipFile(str(target_path)) as zf:
            zf.extractall(path=data_path)

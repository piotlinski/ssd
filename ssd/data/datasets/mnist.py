"""Multi MNIST dataset."""
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from ssd.data.datasets.base import BaseDataset, DataTransformType, TargetTransformType


class MultiScaleMNIST(BaseDataset):
    """Multi-scale MNIST dataset."""

    mnist_keys = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    mnist_url = "http://yann.lecun.com/exdb/mnist/"
    LENGTHS: Dict[str, int] = {"train": 60_000, "test": 10_000}

    def __init__(
        self,
        data_dir: str,
        data_transform: DataTransformType = None,
        target_transform: TargetTransformType = None,
        subset: str = "train",
        image_size: Tuple[int, int, int] = (1, 512, 512),
        digit_size: Tuple[int, int] = (128, 128),
        digit_scales: Tuple[int, ...] = (1, 2, 3),
        max_digits: int = 4,
    ):
        super().__init__(
            str(Path(data_dir).joinpath("mnist")),
            data_transform,
            target_transform,
            subset,
        )
        self.image_size = image_size
        self.digit_size = digit_size
        self.digit_scales = digit_scales
        self.max_digits = max_digits
        self.verify_mnist_dir()
        self.digits = {
            "train": self.load_images("train-images-idx3-ubyte", 60_000),
            "test": self.load_images("t10k-images-idx3-ubyte", 10_000),
        }
        self.labels = {
            "train": self.load_labels("train-labels-idx1-ubyte", 60_000),
            "test": self.load_labels("t10k-labels-idx1-ubyte", 10_000),
        }
        self.annotation: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def __len__(self):
        """Get dataset length."""
        return self.LENGTHS[self.subset]

    def _get_image(self, item: int) -> torch.Tensor:
        n_digits = np.random.randint(1, self.max_digits)
        inds = np.random.choice(
            np.arange(len(self.digits[self.subset])), n_digits, replace=False
        )
        background = np.zeros(self.image_size).astype(np.float32)
        scales = np.random.choice(self.digit_scales, n_digits, replace=True)
        boxes = np.empty((len(inds), 4))
        for box_idx, (idx, scale) in enumerate(zip(inds, scales)):
            x_size, y_size = self.digit_size[0] * scale, self.digit_size[1] * scale
            y_coord = self.random_coordinate(0, self.image_size[1] - y_size)
            x_coord = self.random_coordinate(0, self.image_size[2] - x_size)
            digit = cv2.resize(
                self.digits[self.subset][idx],
                dsize=(x_size, y_size),
                interpolation=cv2.INTER_CUBIC,
            )
            white_ys, white_xs = np.where(digit > 0)
            background[
                0, y_coord : y_coord + y_size, x_coord : x_coord + x_size
            ] += digit
            boxes[box_idx] = [
                x_coord + white_xs.min(),
                y_coord + white_ys.min(),
                x_coord + white_xs.max(),
                y_coord + white_ys.max(),
            ]

        background = np.clip(background, 0, 255)
        background /= 255
        self.annotation = (
            torch.from_numpy(boxes).float(),
            torch.from_numpy(self.labels[self.subset][inds]).float(),
        )
        return torch.from_numpy(background).float()

    def _get_annotation(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.annotation is not None:
            return self.annotation
        raise ValueError

    @staticmethod
    def random_coordinate(min_idx: int, max_idx: int):
        """Sample coordinate in a given range."""
        return np.random.randint(min_idx, max_idx)

    def verify_mnist_dir(self):
        """Check if data already downloaded and invoke downloading if needed."""
        if not all([self.data_dir.joinpath(file).exists() for file in self.mnist_keys]):
            self.data_dir.mkdir()
            self.download()

    def download(self):
        """Download MNIST dataset."""
        for key in self.mnist_keys:
            key += ".gz"
            url = (self.mnist_url + key).format(**locals())
            target_path = self.data_dir.joinpath(key)
            cmd = ["curl", url, "-o", target_path]
            subprocess.call(cmd)
            cmd = ["gunzip", "-d", target_path]
            subprocess.call(cmd)

    def load_images(self, images_file: str, length: int):
        """Load data from image file."""
        with self.data_dir.joinpath(images_file).open() as handle:
            loaded = np.fromfile(file=handle, dtype=np.uint8)
            return loaded[16:].reshape((length, 28, 28, 1))

    def load_labels(self, labels_file: str, length: int):
        """Load data from labels file."""
        with self.data_dir.joinpath(labels_file).open() as handle:
            loaded = np.fromfile(file=handle, dtype=np.uint8)
            return loaded[8:].reshape(length)

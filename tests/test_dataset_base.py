"""Tests for base dataset class."""
from pathlib import Path

import pytest
import torch

from pytorch_ssd.data.datasets.base import BaseDataset, onehot_labels


def test_base_dataset_params():
    """Test base dataset params."""
    path = "."
    ds = BaseDataset(path)
    assert ds.data_dir == Path(path)
    assert ds.data_transform is None
    assert ds.target_transform is None
    assert ds.CLASS_LABELS == []
    assert ds.OBJECT_LABEL == ""


@pytest.mark.parametrize(
    "flat, encoded, n_classes",
    [
        (
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
            3,
        ),
        (
            torch.tensor([[1, 3, 2], [2, 4, 0], [0, 0, 1]]),
            torch.tensor(
                [
                    [
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
            5,
        ),
    ],
)
def test_onehot_encoding(flat, encoded, n_classes):
    """Verify if labels vector is one-hot encoded correctly."""
    assert (onehot_labels(flat, n_classes=n_classes) == encoded).all()

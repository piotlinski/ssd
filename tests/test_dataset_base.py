"""Tests for base dataset class."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ssd.data.datasets.base import BaseDataset


def test_base_dataset_params():
    """Test base dataset params."""
    path = "."
    ds = BaseDataset(path)
    assert ds.data_dir == Path(path)
    assert ds.data_transform is None
    assert ds.target_transform is None


@patch("ssd.data.datasets.base.BaseDataset.__len__", return_value=10)
@patch("ssd.data.datasets.base.BaseDataset.__getitem__")
@pytest.mark.parametrize("channels", (1, 3))
def test_calculating_dataset_stats(getitem_mock, _len_mock, channels):
    """Verify if dataset stats are calculated properly."""
    getitem_mock.return_value = torch.ones((channels, 5, 5)), None, None
    ds = BaseDataset(".")
    pixel_mean, pixel_std = ds.pixel_mean_std()
    assert pixel_mean == channels * (1,)
    assert pixel_std == channels * (0,)

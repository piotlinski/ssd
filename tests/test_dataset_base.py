"""Tests for base dataset class."""
from pathlib import Path

from ssd.data.datasets.base import BaseDataset


def test_base_dataset_params():
    """Test base dataset params."""
    path = "."
    ds = BaseDataset(path)
    assert ds.data_dir == Path(path)
    assert ds.data_transform is None
    assert ds.target_transform is None

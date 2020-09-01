"""Test Multi MNIST dataset."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ssd.data.datasets.mnist import MultiScaleMNIST


@pytest.fixture
def sample_mnist_data():
    """Generate sample mnist data for mocking."""
    data = {
        "train": {
            "images": np.random.randint(0, 255, (5, 512, 512, 3)),
            "boxes": np.random.randint(0, 512, (5, 12, 4)),
            "labels": np.random.randint(-1, 10, (5, 12)),
        }
    }
    return data


@patch("ssd.data.datasets.mnist.h5py.File")
def test_mnist_dataset_params(h5py_mock, sample_mnist_data):
    """Test MultiScaleMNIST dataset params."""
    h5py_mock.return_value.__enter__.return_value = sample_mnist_data
    path = "."
    ds = MultiScaleMNIST(data_dir=path, subset="train", h5_filename="test")
    assert ds.data_dir == Path(path)
    assert ds.subset == "train"
    assert ds.dataset_file == ds.data_dir.joinpath("test")
    assert len(ds) == 5
    assert len(ds.CLASS_LABELS) == 10
    assert ds.OBJECT_LABEL


@patch("ssd.data.datasets.mnist.h5py.File")
def test_mnist_data_fetching(h5py_mock, sample_mnist_data):
    """Test data fetching from h5 file."""
    h5py_mock.return_value.__enter__.return_value = sample_mnist_data
    ds = MultiScaleMNIST(data_dir=".", subset="train", h5_filename="test")
    inserted_labels = sample_mnist_data["train"]["labels"]
    h5py_mock.return_value = sample_mnist_data
    images, boxes, labels = ds[0]
    wrongs = np.count_nonzero(inserted_labels[0] == -1)
    assert images.shape == (512, 512, 3)
    assert boxes.shape == (12 - wrongs, 4)
    assert labels.shape == (12 - wrongs,)

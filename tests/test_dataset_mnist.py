"""Test Multi MNIST dataset."""
from pathlib import Path
from unittest.mock import patch

import pytest

from ssd.data.datasets.mnist import MultiScaleMNIST


@patch("ssd.data.datasets.mnist.MultiScaleMNIST.load_labels")
@patch("ssd.data.datasets.mnist.MultiScaleMNIST.load_images")
@patch("ssd.data.datasets.mnist.MultiScaleMNIST.verify_mnist_dir")
def test_mnist_dataset_params(_verify_mock, _load_images_mock, _load_labels_mock):
    """Test MultiScaleMNIST dataset params."""
    path = "."
    ds = MultiScaleMNIST(
        data_dir=path,
        subset="train",
        image_size=(1, 300, 300),
        digit_size=(24, 24),
        digit_scales=(1, 2, 3, 4),
        max_digits=10,
    )
    assert ds
    assert ds.data_dir == Path(path).joinpath("mnist")
    assert ds.subset == "train"
    assert ds.image_size == (1, 300, 300)
    assert ds.digit_size == (24, 24)
    assert ds.max_digits == 10


@pytest.mark.parametrize("min_idx, max_idx", [(0, 5), (3, 123), (454, 643)])
def test_random_coord(min_idx, max_idx):
    """Test random coordinate sampling."""
    assert min_idx <= MultiScaleMNIST.random_coordinate(min_idx, max_idx) <= max_idx


@patch("ssd.data.datasets.mnist.MultiScaleMNIST.load_labels")
@patch("ssd.data.datasets.mnist.MultiScaleMNIST.load_images")
@patch("ssd.data.datasets.mnist.MultiScaleMNIST.download")
@patch("ssd.data.datasets.mnist.Path.mkdir")
@patch("ssd.data.datasets.mnist.Path.exists", return_value=False)
def test_mnist_dir_verification(
    _exists_mock, mkdir_mock, download_mock, _load_images_mock, _load_labels_mock
):
    """Test if MNIST data dir is verified."""
    _ = MultiScaleMNIST(data_dir=".")
    mkdir_mock.assert_called()
    download_mock.assert_called()


@patch("ssd.data.datasets.mnist.MultiScaleMNIST.load_labels")
@patch("ssd.data.datasets.mnist.MultiScaleMNIST.load_images")
@patch("ssd.data.datasets.mnist.MultiScaleMNIST.download")
@patch("ssd.data.datasets.mnist.Path.mkdir")
@patch("ssd.data.datasets.mnist.Path.exists", return_value=False)
@pytest.mark.parametrize("subset, length", [("train", 60_000), ("test", 10_000)])
def test_mnist_length(
    _exists_mock,
    _mkdir_mock,
    _download_mock,
    _load_images_mock,
    _load_labels_mock,
    subset,
    length,
):
    """Test dataset length."""
    dataset = MultiScaleMNIST(data_dir=".", subset=subset)
    assert len(dataset) == length

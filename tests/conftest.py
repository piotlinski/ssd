from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def sample_image():
    """Sample torch image of correct shape."""
    return torch.zeros((3, 300, 300))


@pytest.fixture
@patch("pytorch_ssd.data.datasets.mnist.h5py.File")
def ssd_params(_file_mock):
    """Create kwargs for SSD."""
    return {"dataset_name": "MNIST", "data_dir": "test"}


@pytest.fixture
def prior_data():
    """Prepare sample prior data."""
    return {
        "image_size": (300, 300),
        "feature_maps": (4, 2),
        "min_sizes": (2, 1),
        "max_sizes": (3, 2),
        "strides": (2, 1),
        "aspect_ratios": ((2, 3), (2,)),
    }

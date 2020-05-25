"""Test dataset loaders."""
from unittest.mock import MagicMock, patch

import pytest

from ssd.data.loaders import DefaultDataLoader, TestDataLoader, TrainDataLoader


@pytest.mark.parametrize("loader", [TrainDataLoader, TestDataLoader])
@patch.dict("ssd.data.loaders.datasets", {"test": MagicMock()})
@patch("ssd.data.loaders.BatchSampler")
@patch("ssd.data.loaders.RandomSampler")
def test_data_loaders(_random_sampler_mock, _batch_sampler_mock, loader, sample_config):
    """Test if data loader is an instance of Torch dataloader."""
    sample_config.DATA.DATASET = "test"
    train_loader = loader(sample_config)
    assert isinstance(train_loader, DefaultDataLoader)

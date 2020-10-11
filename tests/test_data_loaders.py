"""Test dataset loaders."""
from unittest.mock import MagicMock, patch

import pytest

from pyssd.data.loaders import DefaultDataLoader, EvalDataLoader, TrainDataLoader


@pytest.mark.parametrize("loader", [TrainDataLoader, EvalDataLoader])
@patch.dict("pyssd.data.loaders.datasets", {"test": MagicMock()})
@patch("pyssd.data.loaders.BatchSampler")
@patch("pyssd.data.loaders.RandomSampler")
def test_data_loaders(_random_sampler_mock, _batch_sampler_mock, loader, sample_config):
    """Test if data loader is an instance of Torch dataloader."""
    sample_config.DATA.DATASET = "test"
    train_loader = loader(sample_config)
    assert isinstance(train_loader, DefaultDataLoader)

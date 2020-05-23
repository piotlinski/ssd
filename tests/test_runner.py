"""Test SSD runner."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from ssd.run import Runner


def sample_data_loader():
    """Get mock data loader for testing."""
    for _ in range(1):
        yield (
            torch.rand((2, 3, 300, 300)),
            torch.rand((2, 8732, 4)),
            torch.randint(0, 10, (2, 8732)),
        )


@patch("ssd.run.TestDataLoader")
@patch("ssd.run.TrainDataLoader")
@patch("ssd.run.torch.cuda.is_available", return_value=False)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runner_device_cpu(
    _cuda_mock, _train_loader_mock, _test_loader_mock, device, sample_config
):
    """Test if SSD Runner device is initialized correctly with cpu."""
    sample_config.RUNNER.DEVICE = device
    runner = Runner(sample_config)
    assert runner.set_device() == torch.device("cpu")


@patch("ssd.run.TestDataLoader")
@patch("ssd.run.TrainDataLoader")
@patch("ssd.run.torch.cuda.is_available", return_value=True)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runner_device_gpu(
    _cuda_mock, _train_loader_mock, _test_loader_mock, device, sample_config
):
    """Test if SSD Runner device is initialized correctly with gpu."""
    runner = Runner(sample_config)
    runner.config.RUNNER.DEVICE = device
    assert runner.set_device() == torch.device(device)


@patch("ssd.run.TestDataLoader")
@patch("ssd.run.TrainDataLoader")
def test_runner_train(_train_loader_mock, _test_loader_mock, sample_config):
    """Test training SSD model."""
    runner = Runner(sample_config)
    untrained_model = deepcopy(runner.model)
    runner.data_loader = sample_data_loader()
    runner.train()
    assert runner.model != untrained_model

"""Test SSD runner."""
from unittest.mock import patch

import pytest
import torch

from ssd.run import Runner


@patch("ssd.run.torch.cuda.is_available", return_value=False)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runner_device_cpu(_cuda_mock, device, sample_config):
    """Test if SSD Runner device is initialized correctly with cpu."""
    sample_config.RUNNER.DEVICE = device
    runner = Runner(sample_config)
    assert runner.set_device() == torch.device("cpu")


@patch("ssd.run.torch.cuda.is_available", return_value=True)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runner_device_gpu(_cuda_mock, device, sample_config):
    """Test if SSD Runner device is initialized correctly with gpu."""
    runner = Runner(sample_config)
    runner.config.RUNNER.DEVICE = device
    assert runner.set_device() == torch.device(device)

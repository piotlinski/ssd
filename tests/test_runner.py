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


def are_same(model_1, model_2):
    """Compare two models paramaters."""
    for parameter_1, parameter_2 in zip(model_1.parameters(), model_2.parameters()):
        if parameter_1.data.ne(parameter_2.data).sum() > 0:
            return False
    return True


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
@patch("ssd.run.TrainDataLoader", return_value=sample_data_loader())
def test_runner_train(_train_loader_mock, _test_loader_mock, sample_config):
    """Test training SSD model."""
    runner = Runner(sample_config)
    untrained_model = deepcopy(runner.model)
    runner.train()
    assert not are_same(runner.model, untrained_model)


@patch("ssd.run.TestDataLoader", return_value=sample_data_loader())
def test_runner_eval(_test_loader_mock, sample_config):
    """Test evaluating SSD model."""
    runner = Runner(sample_config)
    untrained_model = deepcopy(runner.model)
    runner.eval()
    assert are_same(runner.model, untrained_model)


@pytest.mark.parametrize("data_length", [1, 2])
def test_model_prediction(data_length, sample_config):
    """Test predicting with SSD model."""
    runner = Runner(sample_config)
    sample_inputs = torch.rand((data_length, 3, 300, 300))
    result = runner.predict(inputs=sample_inputs)
    assert len(result) == data_length
    boxes, scores, labels = result[0]
    assert boxes.shape == (sample_config.MODEL.MAX_PER_IMAGE, 4)
    assert scores.shape == (sample_config.MODEL.MAX_PER_IMAGE,)
    assert labels.shape == (sample_config.MODEL.MAX_PER_IMAGE,)

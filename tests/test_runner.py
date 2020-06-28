"""Test SSD runner."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from ssd.run import PlateauWarmUpLRScheduler, Runner


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


@pytest.fixture
def sample_optimizer():
    """Return example optimizer."""
    param_1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))
    param_2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))
    return torch.optim.SGD(
        [{"params": [param_1]}, {"params": [param_2], "lr": 0.1}], lr=0.5
    )


@patch("ssd.run.CheckPointer")
@patch("ssd.run.TestDataLoader")
@patch("ssd.run.TrainDataLoader")
@patch("ssd.run.torch.cuda.is_available", return_value=False)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runner_device_cpu(
    _cuda_mock,
    _train_loader_mock,
    _test_loader_mock,
    _checkpointer_mock,
    device,
    sample_config,
):
    """Test if SSD Runner device is initialized correctly with cpu."""
    sample_config.RUNNER.DEVICE = device
    runner = Runner(sample_config)
    assert runner.set_device() == torch.device("cpu")


@patch("ssd.run.CheckPointer")
@patch("ssd.run.TestDataLoader")
@patch("ssd.run.TrainDataLoader")
@patch("ssd.run.torch.cuda.is_available", return_value=True)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runner_device_gpu(
    _cuda_mock,
    _train_loader_mock,
    _test_loader_mock,
    _checkpointer_mock,
    device,
    sample_config,
):
    """Test if SSD Runner device is initialized correctly with gpu."""
    runner = Runner(sample_config)
    runner.config.RUNNER.DEVICE = device
    assert runner.set_device() == torch.device(device)


@patch("ssd.run.CheckPointer")
@patch("ssd.run.TestDataLoader")
@patch("ssd.run.TrainDataLoader", return_value=sample_data_loader())
def test_runner_train(
    _train_loader_mock, _test_loader_mock, _checkpointer_mock, sample_config,
):
    """Test training SSD model."""
    runner = Runner(sample_config)
    untrained_model = deepcopy(runner.model)
    runner.train()
    assert not are_same(runner.model, untrained_model)


@patch("ssd.run.CheckPointer")
@patch("ssd.run.TestDataLoader", return_value=sample_data_loader())
def test_runner_eval(_test_loader_mock, _checkpointer_mock, sample_config):
    """Test evaluating SSD model."""
    runner = Runner(sample_config)
    untrained_model = deepcopy(runner.model)
    runner.eval()
    assert are_same(runner.model, untrained_model)


@patch("ssd.run.CheckPointer")
@pytest.mark.parametrize("data_length", [1, 2])
def test_model_prediction(_checkpointer_mock, data_length, sample_config):
    """Test predicting with SSD model."""
    runner = Runner(sample_config)
    sample_inputs = torch.rand((data_length, 3, 300, 300))
    result = runner.predict(inputs=sample_inputs)
    assert len(result) == data_length
    boxes, scores, labels = result[0]
    assert boxes.shape == (sample_config.MODEL.MAX_PER_IMAGE, 4)
    assert scores.shape == (sample_config.MODEL.MAX_PER_IMAGE,)
    assert labels.shape == (sample_config.MODEL.MAX_PER_IMAGE,)


@pytest.mark.parametrize(
    "step, warmup_steps, expected",
    [(1, 200, 0.01), (9, 100, 0.1), (29, 50, 0.6), (120, 50, 1.0)],
)
def test_lr_scheduler_factor(step, warmup_steps, expected):
    """Verify calculating warmup factor."""
    factor = PlateauWarmUpLRScheduler.linear_warmup_factor(
        step=step, warmup_steps=warmup_steps
    )
    assert factor == expected


def test_lr_scheduler_params(sample_optimizer):
    """Verify defining LR scheduler."""
    scheduler = PlateauWarmUpLRScheduler(optimizer=sample_optimizer, warmup_steps=5)
    assert all(warmup_steps == 5 for warmup_steps in scheduler.warmup_params)


def test_lr_warmup(sample_optimizer):
    """Verify if LR is increased on warmup."""
    scheduler = PlateauWarmUpLRScheduler(optimizer=sample_optimizer, warmup_steps=5)
    for step in range(1, 10):
        scheduler.dampen()
        lr = [params["lr"] for params in sample_optimizer.param_groups]
        if step < 6:
            assert lr[0] == pytest.approx(0.5 * step / 5)
            assert lr[1] == pytest.approx(0.1 * step / 5)
        else:
            assert lr[0] == pytest.approx(0.5)
            assert lr[1] == pytest.approx(0.1)

import pytest
from yacs.config import CfgNode

from ssd.config import _C as cfg


@pytest.fixture
def sample_config() -> CfgNode:
    """Return sample config with default values."""
    config = cfg.clone()
    config.MODEL.PRETRAINED_URL = ""
    config.RUNNER.DEVICE = "cpu"
    config.DATA.DATASET = "MultiscaleMNIST"
    config.RUNNER.EPOCHS = 1
    config.RUNNER.BATCH_SIZE = 2
    config.MODEL.MAX_PER_IMAGE = 10
    return config

import pytest
from yacs.config import CfgNode

from ssd.config import _C as cfg


@pytest.fixture
def sample_config() -> CfgNode:
    """Return sample config with default values."""
    config = cfg.clone()
    config.MODEL.PRETRAINED_URL = ""
    return config

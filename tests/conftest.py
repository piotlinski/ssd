import pytest
import torch
from yacs.config import CfgNode

from ssd.config import _C as cfg


@pytest.fixture
def sample_config() -> CfgNode:
    """Return sample config with default values."""
    config = cfg.clone()
    config.MODEL.USE_PRETRAINED = False
    config.MODEL.PRETRAINED_URL = ""
    config.RUNNER.DEVICE = "cpu"
    config.DATA.DATASET = "MultiscaleMNIST"
    config.RUNNER.EPOCHS = 1
    config.RUNNER.BATCH_SIZE = 2
    config.MODEL.MAX_PER_IMAGE = 10
    config.RUNNER.USE_TENSORBOARD = False
    return config


@pytest.fixture
def sample_image(sample_config):
    """Sample torch image of correct shape."""
    return torch.zeros((3, *sample_config.DATA.SHAPE))


@pytest.fixture
def sample_prediction(sample_config):
    """Sample cls_logits and bbox_pred for given config."""
    dim = sum(
        [
            features ** 2 * boxes
            for features, boxes in zip(
                sample_config.DATA.PRIOR.FEATURE_MAPS,
                sample_config.DATA.PRIOR.BOXES_PER_LOC,
            )
        ]
    )
    return torch.zeros((1, dim, sample_config.DATA.N_CLASSES)), torch.zeros((1, dim, 4))

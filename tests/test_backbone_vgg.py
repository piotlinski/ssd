"""Test VGG backbone."""
from collections import Counter

import pytest
import torch
import torch.nn as nn

from pyssd.modeling.backbones.vgg import VGG11, VGG16, L2Norm


def verify_vgg_backbone(backbone: nn.ModuleList, batch_norm: bool):
    """Verify backbone layers in VGG."""
    n_conv2d = 15
    n_relu = 15
    n_batch_norm = 13
    n_maxpool = 5
    assert len(backbone) == n_conv2d + n_relu + batch_norm * n_batch_norm + n_maxpool
    types_count = Counter(map(type, backbone))
    assert types_count[nn.Conv2d] == n_conv2d
    assert types_count[nn.ReLU] == n_relu
    assert types_count[nn.BatchNorm2d] == (n_batch_norm if batch_norm else 0)
    assert types_count[nn.MaxPool2d] == n_maxpool


@pytest.mark.parametrize("n_channels, scale", [(6, 1.0), (16, 0.5), (64, 1.5)])
def test_l2norm(n_channels, scale):
    """Test L2Norm layer"""
    l2_norm = L2Norm(n_channels=n_channels, scale=scale)
    assert l2_norm.n_channels == n_channels
    assert l2_norm.gamma == scale
    assert l2_norm.weight.shape[0] == n_channels

    sample_data = torch.rand((1, n_channels, 4, 4))

    assert l2_norm(sample_data).shape == sample_data.shape


@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("shape, extras_length", [((300, 300), 8), ((512, 512), 10)])
def test_vgg16_defaults(shape, extras_length, batch_norm, sample_config):
    """Verify layers in VGG16 backbone."""
    sample_config.DATA.SHAPE = shape
    sample_config.MODEL.BATCH_NORM = batch_norm
    vgg = VGG16(sample_config)
    verify_vgg_backbone(vgg.backbone, batch_norm=batch_norm)

    assert len(vgg.extras) == extras_length
    assert all([isinstance(layer, nn.Conv2d) for layer in vgg.extras])


@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("backbone", [VGG16, VGG11])
def test_forward(backbone, batch_norm, sample_config):
    """Verify forward function in VGG backbones."""
    sample_config.MODEL.BATCH_NORM = batch_norm
    vgg = backbone(sample_config)
    inputs = torch.rand((1, 3, 300, 300))
    outputs = vgg(inputs)
    assert len(outputs) == len(vgg.out_channels)

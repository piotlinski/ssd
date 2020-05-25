"""Test VGG300 backbone."""
from collections import Counter
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from ssd.modeling.backbones.vgg import VGG300, L2Norm


def verify_vgg300_backbone(backbone: nn.ModuleList, batch_norm: bool):
    """Verify backbone layers in VGG300."""
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


@patch("ssd.modeling.backbones.vgg.nn.Module.load_state_dict")
@patch("ssd.modeling.backbones.vgg.torch.load")
@patch("ssd.modeling.backbones.vgg.cache_url")
def test_downloading_pretrained(
    cache_url_mock, torch_load_mock, load_state_dict_mock, sample_config
):
    sample_config.MODEL.PRETRAINED_URL = "test"
    VGG300(sample_config)
    cache_url_mock.assert_called_with("test")
    torch_load_mock.assert_called_with(cache_url_mock.return_value, map_location="cpu")
    load_state_dict_mock.assert_called_with(torch_load_mock.return_value)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("batch_norm", [True, False])
def test_vgg300_defaults(batch_norm, channels, sample_config):
    """Verify layers in VGG300 backbone."""
    sample_config.MODEL.BATCH_NORM = batch_norm
    sample_config.DATA.CHANNELS = channels
    vgg = VGG300(sample_config)
    verify_vgg300_backbone(vgg.backbone, batch_norm=batch_norm)

    assert len(vgg.extras) == 8
    assert all([isinstance(layer, nn.Conv2d) for layer in vgg.extras])


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("batch_norm", [True, False])
def test_forward(batch_norm, channels, sample_config):
    """Verify forward function in VGG300 backbone."""
    sample_config.MODEL.BATCH_NORM = batch_norm
    sample_config.DATA.CHANNELS = channels
    vgg = VGG300(sample_config)
    inputs = torch.rand((1, channels, 300, 300))
    outputs = vgg(inputs)
    assert len(outputs) == len(vgg.out_channels)

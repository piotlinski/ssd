"""Test VGG backbone."""
import pytest
import torch

from pytorch_ssd.modeling.backbones import MobileNetV2
from pytorch_ssd.modeling.backbones.vgg import (
    VGG300,
    VGG300BN,
    VGG512,
    VGG512BN,
    L2Norm,
    VGGLite,
    VGGLiteBN,
)


@pytest.mark.parametrize("n_channels, scale", [(6, 1.0), (16, 0.5), (64, 1.5)])
def test_l2norm(n_channels, scale):
    """Test L2Norm layer"""
    l2_norm = L2Norm(n_channels=n_channels, scale=scale)
    assert l2_norm.n_channels == n_channels
    assert l2_norm.gamma == scale
    assert l2_norm.weight.shape[0] == n_channels

    sample_data = torch.rand((1, n_channels, 4, 4))

    assert l2_norm(sample_data).shape == sample_data.shape


@pytest.mark.parametrize(
    "backbone, shape",
    [
        (VGG300, 300),
        (VGG300BN, 300),
        (VGG512, 512),
        (VGG512BN, 512),
        (VGGLite, 300),
        (VGGLiteBN, 300),
        (MobileNetV2, 300),
    ],
)
def test_forward(backbone, shape):
    """Verify forward function in VGG backbones."""
    vgg = backbone(use_pretrained=False)
    inputs = torch.rand((1, 3, shape, shape))
    outputs = vgg(inputs)
    assert len(outputs) == len(vgg.out_channels)

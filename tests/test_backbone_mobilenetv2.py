"""Test MobileNetV2 backbone."""
import torch

from pyssd.modeling.backbones import MobileNetV2


def test_forward(sample_config):
    """Verify forward function in MobileNetV2 backbone."""
    backbone = MobileNetV2(config=sample_config)
    inputs = torch.rand((1, 3, 300, 300))
    outputs = backbone(inputs)
    assert len(outputs) == len(backbone.out_channels)

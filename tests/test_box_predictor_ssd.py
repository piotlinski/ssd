"""Test SSD box predictors."""
import torch
from torch import nn

from ssd.modeling.box_predictors.ssd import SSDBoxPredictor


def test_ssd_predictor(sample_config):
    """Test SSD box predictor params."""
    output_channels_list = [4, 8, 4, 2]
    predictor = SSDBoxPredictor(sample_config, output_channels_list)
    assert len(predictor.cls_headers) == 4
    assert len(predictor.reg_headers) == 4
    assert all([isinstance(layer, nn.Conv2d) for layer in predictor.cls_headers])
    assert all([isinstance(layer, nn.Conv2d) for layer in predictor.reg_headers])
    for cls_layer, reg_layer, output_channels in zip(
        predictor.cls_headers, predictor.reg_headers, output_channels_list
    ):
        assert cls_layer.in_channels == output_channels
        assert reg_layer.in_channels == output_channels


def test_ssd_predictor_forward(sample_config):
    """Test forwarding data through SSD box predictor."""
    inputs = torch.rand((1, 1, 4, 1, 1))
    predictor = SSDBoxPredictor(sample_config, [4, 8, 4, 2])
    class_output, bbox_output = predictor(inputs)
    assert class_output.shape[-1] == sample_config.DATA.N_CLASSES
    assert bbox_output.shape[-1] == 4

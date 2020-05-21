"""Test SSD model."""
import torch

from ssd.modeling.backbones import backbones
from ssd.modeling.box_predictors import box_predictors
from ssd.modeling.model import SSD


def test_model_setup(sample_config):
    """Test model building."""
    model = SSD(sample_config)
    assert isinstance(model.backbone, backbones[sample_config.MODEL.BACKBONE])
    assert isinstance(
        model.predictor, box_predictors[sample_config.MODEL.BOX_PREDICTOR]
    )


def test_model_forward(sample_config):
    """Test model forward method."""
    model = SSD(sample_config)
    data = torch.rand((1, 3, 300, 300))
    cls_logits, bbox_pred = model(data)
    assert cls_logits.shape[0] == 1
    assert cls_logits.shape[2] == sample_config.DATA.N_CLASSES
    assert bbox_pred.shape[0] == 1
    assert bbox_pred.shape[2] == 4
    assert cls_logits.shape[1] == bbox_pred.shape[1]

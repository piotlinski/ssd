"""Test SSD model."""
import pytest
import torch

from pyssd.modeling.backbones import backbones
from pyssd.modeling.box_predictors import box_predictors
from pyssd.modeling.model import SSD, process_model_output


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


@pytest.mark.parametrize("n_predictions", [0, 1, 2])
def test_model_output_processing(n_predictions):
    """Test processing model output for using it."""
    cls_logits = torch.rand(n_predictions, 2, 3)
    bbox_pred = torch.rand(n_predictions, 2, 4)
    processed = list(
        process_model_output(
            detections=(cls_logits, bbox_pred),
            image_size=(300, 300),
            confidence_threshold=0.1,
            nms_threshold=0.5,
            max_per_image=100,
        )
    )
    assert 0 <= len(processed) <= n_predictions

"""Test SSD model."""
import pytest
import torch

from pyssd.modeling.backbones import backbones
from pyssd.modeling.box_predictors import box_predictors
from pyssd.modeling.model import SSD


def test_model_setup(ssd_params):
    """Test model building."""
    model = SSD(**ssd_params)
    assert isinstance(model.backbone, backbones[ssd_params["backbone_name"]])
    assert isinstance(model.predictor, box_predictors[ssd_params["predictor_name"]])
    assert model.class_labels == ssd_params["class_labels"]


def test_model_forward(ssd_params):
    """Test model forward method."""
    model = SSD(**ssd_params)
    data = torch.rand((1, 3, 300, 300))
    cls_logits, bbox_pred = model(data)
    assert cls_logits.shape[0] == 1
    assert cls_logits.shape[2] == ssd_params["n_classes"]
    assert bbox_pred.shape[0] == 1
    assert bbox_pred.shape[2] == 4
    assert cls_logits.shape[1] == bbox_pred.shape[1]


@pytest.mark.parametrize("n_predictions", [0, 1, 2])
def test_model_output_processing(n_predictions, ssd_params):
    """Test processing model output for using it."""
    model = SSD(**ssd_params)
    cls_logits = torch.rand(n_predictions, 2, 3)
    bbox_pred = torch.rand(n_predictions, 2, 4)
    processed = list(
        model.process_model_output(
            detections=(cls_logits, bbox_pred), confidence_threshold=0.1
        )
    )
    assert 0 <= len(processed) <= n_predictions


def test_model_prediction_processing(ssd_params):
    """Verify processing model prediction."""
    model = SSD(**ssd_params)
    data = torch.rand((1, 3, 300, 300))
    cls_logits, bbox_pred = model(data)
    detections = model.process_model_prediction(
        cls_logits=cls_logits, bbox_pred=bbox_pred
    )
    assert detections

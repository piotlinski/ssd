"""Test SSD model."""
import pytest
import torch

from ssd.modeling.backbones import backbones
from ssd.modeling.box_predictors import box_predictors
from ssd.modeling.model import SSD, non_max_suppression, process_model_output


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


@pytest.mark.parametrize("n_boxes", [1, 2, 3, 4])
def test_nms(n_boxes):
    """Test non-max suppression return format."""
    boxes = torch.rand((n_boxes, 4))
    scores = torch.rand(n_boxes)
    indices = torch.rand(n_boxes)
    threshold = 0.5
    nms = non_max_suppression(
        boxes=boxes, scores=scores, indices=indices, threshold=threshold
    )
    assert 0 <= nms.shape[0] <= n_boxes


def test_nms_empty():
    """Test non-max suppression for empty tensor."""
    empty = torch.empty(0)
    nms = non_max_suppression(boxes=empty, scores=empty, indices=empty, threshold=0.0)
    assert nms.numel() == 0


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

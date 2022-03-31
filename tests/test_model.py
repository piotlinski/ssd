"""Test SSD model."""
import pytest
import torch

from pytorch_ssd.modeling.model import SSD


def test_model_setup(ssd_params):
    """Test model building."""
    model = SSD(**ssd_params)
    assert model


def test_model_forward(ssd_params):
    """Test model forward method."""
    model = SSD(**ssd_params)
    data = torch.rand((1, 3, 300, 300))
    detections = model(data)
    assert detections


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

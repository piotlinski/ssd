import pytest
import torch


@pytest.fixture
def sample_image():
    """Sample torch image of correct shape."""
    return torch.zeros((3, 300, 300))


@pytest.fixture
def ssd_params():
    """Create kwargs for SSD."""
    return {
        "backbone_name": "VGG300",
        "use_pretrained_backbone": False,
        "predictor_name": "SSD",
        "image_size": (300, 300),
        "n_classes": 11,
        "center_variance": 0.2,
        "size_variance": 0.1,
        "iou_threshold": 0.5,
        "confidence_threshold": 0.8,
        "nms_threshold": 0.5,
        "max_per_image": 100,
        "class_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "object_label": "digit",
    }


@pytest.fixture
def prior_data():
    """Prepare sample prior data."""
    return {
        "image_size": (300, 300),
        "feature_maps": (4, 2),
        "min_sizes": (2, 1),
        "max_sizes": (3, 2),
        "strides": (2, 1),
        "aspect_ratios": ((2, 3), (2,)),
    }

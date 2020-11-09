"""Test multi box loss function."""
import pytest
import torch

from pyssd.modeling.loss import MultiBoxLoss


@pytest.fixture
def loss():
    """Return loss module."""
    return MultiBoxLoss(negative_positive_ratio=0.3)


@pytest.fixture
def class_prediction():
    """Example class prediction."""
    return torch.rand(1, 3, 3)


@pytest.fixture
def box_prediction():
    """Example bounding box prediction."""
    return torch.rand(1, 3, 4)


@pytest.fixture
def labels():
    """Example labels."""
    return torch.randint(1, 3, (1, 3))


@pytest.fixture
def box_ground_truth():
    """Example bounding box ground truth."""
    return torch.rand(1, 3, 4)


def test_classification_loss(loss, class_prediction, labels):
    """Test classification loss."""
    classification_loss = loss.classification_loss(class_prediction, labels)
    assert classification_loss.item()


def test_box_regression_loss(loss, box_prediction, box_ground_truth):
    """Test box regression loss."""
    mask = torch.ones_like(box_prediction[..., 0]) == 1
    regression_loss = loss.box_regression_loss(box_prediction, box_ground_truth, mask)
    assert regression_loss.item()


def test_total_loss(loss, class_prediction, box_prediction, labels, box_ground_truth):
    """Test both loss components calculation."""
    regression_loss, classification_loss = loss(
        class_prediction, box_prediction, labels, box_ground_truth
    )
    assert regression_loss.item()
    assert classification_loss.item()

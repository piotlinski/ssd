"""Test data transforms."""
from unittest.mock import MagicMock

import pytest
import torch

from pyssd.data.priors import process_prior
from pyssd.data.transforms import DataTransform, SSDTargetTransform, TrainDataTransform


@pytest.fixture
def anchors(prior_data):
    """Sample anchors for target transform."""
    return process_prior(**prior_data)


@pytest.fixture
def inputs():
    """Sample inputs to transforms."""
    image = torch.zeros(100, 100, 3)
    bboxes = torch.tensor(
        [[10.0, 30.0, 70.0, 70.0], [0.0, 0.0, 50.0, 50.0]], dtype=torch.float32
    )
    labels = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return image, bboxes, labels


@pytest.mark.parametrize("n_classes", [2, 10])
def test_ssd_target_transform(n_classes, anchors, inputs):
    """Test SSD target transform."""
    transform = SSDTargetTransform(
        anchors=anchors,
        image_size=(300, 300),
        n_classes=n_classes,
        center_variance=0.1,
        size_variance=0.2,
        iou_threshold=0.5,
    )
    _, gt_boxes, gt_labels = inputs

    locations, labels = transform(gt_boxes.numpy(), gt_labels.numpy())
    assert locations.shape[-1] == 4
    assert labels.shape[0] == locations.shape[0]
    if n_classes == 2:
        assert all(labels.unique() == torch.tensor([0.0, 1.0]))


def test_data_transform(inputs):
    """Test basic data transform."""
    image_size = (20, 20)
    transform = DataTransform(
        image_size=image_size, pixel_mean=(0.0, 0.0, 0.0), pixel_std=(1.0, 1.0, 1.0)
    )
    image, bboxes, labels = inputs
    t_image, t_bboxes, t_labels = transform(image=image, bboxes=bboxes, labels=labels)
    assert t_image.shape == (3, *image_size)
    assert (torch.round(100 * t_bboxes) == bboxes).all()
    assert (t_labels == labels).all()


def test_adding_transforms(inputs):
    """Test adding transforms to basic data transform."""
    transform_mock = MagicMock()
    transform = DataTransform(
        image_size=(30, 30),
        pixel_mean=(0.0, 0.0, 0.0),
        pixel_std=(1.0, 1.0, 1.0),
        transforms=[transform_mock],
    )
    assert transform_mock in transform.transforms


def test_image_only_transform(inputs):
    """Test transforming only image."""
    image_size = (20, 20)
    transform = DataTransform(
        image_size=image_size, pixel_mean=(0.0, 0.0, 0.0), pixel_std=(1.0, 1.0, 1.0)
    )
    image, *_ = inputs
    t_image, t_bboxes, t_labels = transform(image=image)
    assert t_image.shape == (3, *image_size)
    assert t_bboxes is None
    assert t_labels is None


def test_train_transform(inputs):
    """Test train data transform."""
    image_size = (20, 20)
    transform = TrainDataTransform(
        image_size=image_size, pixel_mean=(0.0, 0.0, 0.0), pixel_std=(1.0, 1.0, 1.0)
    )
    image, bboxes, labels = inputs
    t_image, _, _ = transform(image=image, bboxes=bboxes, labels=labels)
    assert t_image.shape == (3, *image_size)

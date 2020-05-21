"""Test data transforms."""
from unittest.mock import MagicMock

import numpy as np
import torch

from ssd.data.transforms import DataTransform, SSDTargetTransform, TrainDataTransform


def test_ssd_target_transform(sample_config):
    """Test SSD target transform."""
    transform = SSDTargetTransform(sample_config)
    gt_boxes = np.array(
        [[10.0, 30.0, 70.0, 70.0], [0.0, 0.0, 50.0, 50.0]], dtype=np.float32
    )
    gt_labels = np.array([1.0, 2.0], dtype=np.float32)

    locations, labels = transform(gt_boxes, gt_labels)
    assert locations.shape[-1] == 4
    assert labels.shape[0] == locations.shape[0]


def test_data_transform(sample_config):
    """Test basic data transform."""
    transform = DataTransform(sample_config)
    image = torch.zeros(3, 10, 10)
    bboxes = torch.tensor([[0.0, 2.0, 4.0, 6.0], [5.0, 5.0, 7.0, 7.0]])
    labels = torch.tensor([1, 2])
    t_image, t_bboxes, t_labels = transform(image=image, bboxes=bboxes, labels=labels)
    assert t_image.shape == (sample_config.DATA.CHANNELS, *sample_config.DATA.SHAPE)
    assert (torch.round(10 * t_bboxes) == bboxes).all()
    assert (t_labels == labels).all()


def test_adding_transforms(sample_config):
    """Test adding transforms to basic data transform."""
    transform_mock = MagicMock()
    transform = DataTransform(sample_config, [transform_mock])
    assert transform_mock in transform.transforms


def test_image_only_transform(sample_config):
    """Test transforming only image."""
    transform = DataTransform(sample_config)
    image = torch.zeros(3, 10, 10)
    t_image, t_bboxes, t_labels = transform(image=image)
    assert t_image.shape == (sample_config.DATA.CHANNELS, *sample_config.DATA.SHAPE)
    assert t_bboxes is None
    assert t_labels is None


def test_train_transform(sample_config):
    """Test train data transform."""
    transform = TrainDataTransform(sample_config)
    image = torch.zeros(3, 10, 10)
    t_image, _, _ = transform(image=image)
    assert t_image.shape == (sample_config.DATA.CHANNELS, *sample_config.DATA.SHAPE)

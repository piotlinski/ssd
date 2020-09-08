"""Test SSD bounding box utils."""
from math import e

import pytest
import torch

from pyssd.data.bboxes import (
    area,
    assign_priors,
    center_bbox_to_corner_bbox,
    convert_boxes_to_locations,
    convert_locations_to_boxes,
    corner_bbox_to_center_bbox,
    iou,
)


@pytest.mark.parametrize(
    "locations, bounding_boxes",
    [
        (
            torch.tensor([[[0.0, 0.0, 0.0, 0.0]]]),
            torch.tensor([[[1.0, 1.0, 1.0, 1.0]]]),
        ),
        (
            torch.tensor([[[1.0, 1.0, 2.0, 2.0]], [[-1.0, -1.0, 2.0, 3.0]]]),
            torch.tensor(
                [[[2.0, 2.0, e ** 2.0, e ** 2.0]], [[0.0, 0.0, e ** 2.0, e ** 3.0]]]
            ),
        ),
    ],
)
def test_location_to_bbox(locations, bounding_boxes):
    """Test converting location tensor to bounding boxes."""
    priors = torch.tensor([[1, 1, 1, 1]])  # squeezed
    converted = convert_locations_to_boxes(
        locations, priors, center_variance=1, size_variance=1
    )
    assert (torch.round(converted) == torch.round(bounding_boxes)).all()


@pytest.mark.parametrize(
    "bounding_boxes, locations",
    [
        (
            torch.tensor([[[1.0, 1.0, 1.0, 1.0]]]),
            torch.tensor([[[0.0, 0.0, 0.0, 0.0]]]),
        ),
        (
            torch.tensor(
                [[[2.0, 2.0, e ** 2.0, e ** 2.0]], [[0.0, 0.0, e ** 2.0, e ** 3.0]]]
            ),
            torch.tensor([[[1.0, 1.0, 2.0, 2.0]], [[-1.0, -1.0, 2.0, 3.0]]]),
        ),
    ],
)
def test_bbox_to_location(bounding_boxes, locations):
    """Test converting bounding boxes to location tensor."""
    priors = torch.tensor([[1, 1, 1, 1]])  # squeezed
    converted = convert_boxes_to_locations(
        bounding_boxes, priors, center_variance=1, size_variance=1
    )
    assert (torch.round(100 * converted) == torch.round(100 * locations)).all()


@pytest.mark.parametrize(
    "bounding_boxes",
    [
        torch.tensor([[[1.0, 1.0, 1.0, 1.0]]]),
        torch.tensor(
            [[[2.0, 2.0, e ** 2.0, e ** 2.0]], [[0.0, 0.0, e ** 2.0, e ** 3.0]]]
        ),
    ],
)
def test_bbox_location_reversible(bounding_boxes):
    """Test if converting bounding boxes to location tensor is reversible."""
    priors = torch.tensor([[1, 1, 1, 1]])  # squeezed
    converted = convert_boxes_to_locations(
        bounding_boxes, priors, center_variance=1, size_variance=1
    )
    deconverted = convert_locations_to_boxes(
        converted, priors, center_variance=1, size_variance=1
    )
    assert (deconverted == bounding_boxes).all()


@pytest.mark.parametrize(
    "locations",
    [
        torch.tensor([[[0.0, 0.0, 0.0, 0.0]]]),
        torch.tensor(
            [[[2.0, 2.0, e ** 2.0, e ** 2.0]], [[0.0, 0.0, e ** 2.0, e ** 3.0]]]
        ),
    ],
)
def test_location_bbox_reversible(locations):
    """Test if converting location tensor to bounding boxes is reversible."""
    priors = torch.tensor([[1, 1, 1, 1]])  # squeezed
    converted = convert_locations_to_boxes(
        locations, priors, center_variance=1, size_variance=1
    )
    deconverted = convert_boxes_to_locations(
        converted, priors, center_variance=1, size_variance=1
    )
    assert (deconverted == locations).all()


@pytest.mark.parametrize(
    "center_bbox, corner_bbox",
    [
        (torch.tensor([5.0, 5.0, 10.0, 10.0]), torch.tensor([0.0, 0.0, 10.0, 10.0])),
        (
            torch.tensor([[3.0, 4.0, 2.0, 2.0], [8.0, 5.0, 4.0, 2.0]]),
            torch.tensor([[2.0, 3.0, 4.0, 5.0], [6.0, 4.0, 10.0, 6.0]]),
        ),
    ],
)
def test_center_to_corner(center_bbox, corner_bbox):
    """Test converting bbox from center- to corner-form."""
    converted = center_bbox_to_corner_bbox(center_bbox)
    assert (converted == corner_bbox).all()


@pytest.mark.parametrize(
    "corner_bbox, center_bbox",
    [
        (torch.tensor([0.0, 0.0, 10.0, 10.0]), torch.tensor([5.0, 5.0, 10.0, 10.0])),
        (
            torch.tensor([[2.0, 3.0, 4.0, 5.0], [6.0, 4.0, 10.0, 6.0]]),
            torch.tensor([[3.0, 4.0, 2.0, 2.0], [8.0, 5.0, 4.0, 2.0]]),
        ),
    ],
)
def test_corner_to_center(corner_bbox, center_bbox):
    """Test converting bbox from corner- to center-form."""
    converted = corner_bbox_to_center_bbox(corner_bbox)
    assert (converted == center_bbox).all()


@pytest.mark.parametrize(
    "center_bbox",
    [
        torch.tensor([5.0, 5.0, 10.0, 10.0]),
        torch.tensor([[3.0, 4.0, 2.0, 2.0], [8.0, 5.0, 4.0, 2.0]]),
    ],
)
def test_center_corner_reversible(center_bbox):
    """Test if converting from center to corner form is reversible."""
    converted = center_bbox_to_corner_bbox(center_bbox)
    deconverted = corner_bbox_to_center_bbox(converted)
    assert (deconverted == center_bbox).all()


@pytest.mark.parametrize(
    "corner_bbox",
    [
        torch.tensor([0.0, 0.0, 10.0, 10.0]),
        torch.tensor([[2.0, 3.0, 4.0, 5.0], [6.0, 4.0, 10.0, 6.0]]),
    ],
)
def test_corner_center_reversible(corner_bbox):
    """Test if converting from corner to center form is reversible."""
    converted = corner_bbox_to_center_bbox(corner_bbox)
    deconverted = center_bbox_to_corner_bbox(converted)
    assert (deconverted == corner_bbox).all()


@pytest.mark.parametrize(
    "left_top, right_bottom, expected_area",
    [([[1, 1]], [[3, 3]], [4]), ([[0, 3], [5, 8]], [[6, 9], [6, 9]], [36, 1])],
)
def test_area(left_top, right_bottom, expected_area):
    """Test area calculation."""
    left_top = torch.tensor(left_top)
    right_bottom = torch.tensor(right_bottom)
    expected_area = torch.tensor(expected_area)
    assert (area(left_top, right_bottom) == expected_area).all()


@pytest.mark.parametrize(
    "boxes_1, boxes_2, expected_iou",
    [
        ([[0, 0, 10, 10]], [[0, 0, 5, 5]], [0.25]),
        (
            [[0, 0, 10, 10], [0, 0, 20, 20]],
            [[0, 10, 10, 20], [0, 0, 20, 20]],
            [0.0, 1.0],
        ),
    ],
)
def test_iou(boxes_1, boxes_2, expected_iou):
    """Test IOU calculation."""
    boxes_1 = torch.tensor(boxes_1)
    boxes_2 = torch.tensor(boxes_2)
    expected_iou = torch.tensor(expected_iou)
    assert (
        torch.round(100 * iou(boxes_1, boxes_2)) == torch.round(100 * expected_iou)
    ).all()


def test_prior_assigning():
    """Test assigning bboxes to prior."""
    gt_boxes = torch.tensor([[1, 3, 7, 7], [0, 0, 12, 12]])
    gt_labels = torch.tensor([1, 2])
    corner_form_priors = torch.tensor([[0, 0, 10, 10], [3, 3, 7, 7]])
    iou_threshold = 0.1
    boxes, labels = assign_priors(
        gt_boxes, gt_labels, corner_form_priors, iou_threshold
    )
    assert (boxes[0] == gt_boxes[1]).all()
    assert (boxes[1] == gt_boxes[0]).all()
    assert labels[0] == gt_labels[1]
    assert labels[1] == gt_labels[0]

"""Test SSD bounding box utils."""
from math import e

import pytest
import torch

from ssd.dataset.bboxes import convert_boxes_to_locations, convert_locations_to_boxes


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

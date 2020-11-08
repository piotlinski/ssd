"""Test SSD data priors utils."""
import pytest

from pyssd.data.priors import (
    all_prior_boxes,
    feature_map_prior_boxes,
    prior_boxes,
    process_prior,
    rect_box,
    square_box,
    unit_center,
)


@pytest.mark.parametrize(
    "indices, image_size, stride, expected",
    [
        ((0, 0), (100, 100), 100, (0.5, 0.5)),
        ((3, 4), (140, 90), 10, (0.25, 0.5)),
        ((43, 21), (87, 215), 4, (2, 0.4)),
    ],
)
def test_unit_center(indices, image_size, stride, expected):
    """Test cell center calculation."""
    assert unit_center(indices, image_size, stride) == expected


@pytest.mark.parametrize(
    "box_size, image_size, expected",
    [
        (5, (10, 10), (0.5, 0.5)),
        (30, (240, 480), (0.0625, 0.125)),
        (27, (270, 540), (0.05, 0.1)),
    ],
)
def test_square_box(box_size, image_size, expected):
    """Test square box size calculation."""
    assert square_box(box_size, image_size) == expected


@pytest.mark.parametrize(
    "box_size, image_size, ratio, expected",
    [
        (5, (10, 10), 4, ((1.0, 0.25), (0.25, 1.0))),
        (30, (240, 480), 25, ((0.3125, 0.025), (0.0125, 0.625))),
        (27, (270, 540), 0.25, ((0.025, 0.2), (0.1, 0.05))),
    ],
)
def test_rect_box(box_size, image_size, ratio, expected):
    """Test rectangular box size calculation."""
    assert rect_box(box_size, image_size, ratio) == expected


@pytest.mark.parametrize(
    "rect_ratios, expected_length",
    [(tuple(), 2), ((2,), 4), ((2, 3), 6), ((2, 3, 4), 8)],
)
def test_prior_boxes(rect_ratios, expected_length):
    """Test per-cell prior boxes."""
    boxes = list(
        prior_boxes(
            image_size=(300, 300),
            indices=(0, 0),
            small_size=2,
            big_size=10,
            stride=5,
            rect_ratios=rect_ratios,
        )
    )
    assert len(boxes) == expected_length


@pytest.mark.parametrize(
    "feature_map, rect_ratios, expected_length",
    [(10, (2,), 400), (3, (2, 3), 54), (2, (2, 3, 4), 32)],
)
def test_feature_map_prior_boxes(feature_map, rect_ratios, expected_length):
    """Test per-feature map prior boxes."""
    boxes = list(
        feature_map_prior_boxes(
            image_size=(300, 300),
            feature_map=feature_map,
            small_size=2,
            big_size=10,
            stride=5,
            rect_ratios=rect_ratios,
        )
    )
    assert len(boxes) == expected_length


def test_all_prior_boxes(prior_data):
    """Test generating all prior boxes."""
    all_boxes = list(all_prior_boxes(**prior_data))
    assert len(all_boxes) == 112


def test_prior_processing(prior_data):
    """Test generating prior tensor."""
    prior = process_prior(**prior_data)
    assert (0 <= prior).all()
    assert (prior <= 1).all()
    assert prior.shape[0] == 112
    assert prior.shape[1] == 4
    assert len(prior.shape) == 2

"""Test SSD data priors utils."""
import pytest

from ssd.data.priors import (
    all_prior_boxes,
    feature_map_prior_boxes,
    prior_boxes,
    process_prior,
    rect_box,
    square_box,
    unit_center,
    unit_scale,
)


@pytest.fixture
def prior_data():
    """Prepare sample prior data."""
    image_size = (300, 300)
    feature_maps = (4, 2)
    min_sizes = (2, 1)
    max_sizes = (3, 2)
    strides = (2, 1)
    aspect_ratios = ((2, 3), (2,))
    return {
        "image_size": image_size,
        "feature_maps": feature_maps,
        "min_sizes": min_sizes,
        "max_sizes": max_sizes,
        "strides": strides,
        "aspect_ratios": aspect_ratios,
    }


@pytest.mark.parametrize(
    "image_size, stride, expected",
    [
        ((100, 100), 5, (20, 20)),
        ((300, 250), 10, (30, 25)),
        ((550, 770), 220, (2.5, 3.5)),
    ],
)
def test_unit_scale(image_size, stride, expected):
    """Test scale calculation."""
    assert unit_scale(image_size, stride) == expected


@pytest.mark.parametrize(
    "indices, scale, expected",
    [
        ((0, 0), (1, 1), (0.5, 0.5)),
        ((3, 4), (14, 9), (0.25, 0.5)),
        ((43, 21), (21.75, 53.75), (2, 0.4)),
    ],
)
def test_unit_center(indices, scale, expected):
    """Test cell center calculation."""
    assert unit_center(indices, scale) == expected


@pytest.mark.parametrize(
    "box_size, scale, expected",
    [(5, (0.5, 0.5), (10, 10)), (15, (2, 5), (3, 7.5)), (27, (0.4, 3), (9, 67.5))],
)
def test_square_box(box_size, scale, expected):
    """Test square box size calculation."""
    assert square_box(box_size, scale) == expected


@pytest.mark.parametrize(
    "box_size, scale, ratio, expected",
    [
        (5, (0.5, 0.5), 4, ((20, 5), (5, 20))),
        (15, (2, 5), 9, ((9, 2.5), (1, 22.5))),
        (27, (0.4, 3), 0.25, ((4.5, 135), (18, 33.75))),
    ],
)
def test_rect_box(box_size, scale, ratio, expected):
    """Test rectangular box size calculation."""
    assert rect_box(box_size, scale, ratio) == expected


@pytest.mark.parametrize(
    "rect_ratios, expected_length",
    [(tuple(), 2), ((2,), 4), ((2, 3), 6), ((2, 3, 4), 8)],
)
def test_prior_boxes(rect_ratios, expected_length):
    """Test per-cell prior boxes."""
    boxes = list(
        prior_boxes(
            indices=(0, 0),
            scale=(1.0, 1.0),
            small_size=2,
            big_size=10,
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
            feature_map,
            scale=(1.0, 1.0),
            small_size=2,
            big_size=10,
            rect_ratios=rect_ratios,
        )
    )
    assert len(boxes) == expected_length


def test_all_prior_boxes(prior_data):
    """Test generating all prior boxes."""
    all_boxes = list(all_prior_boxes(**prior_data))
    assert len(all_boxes) == 112


@pytest.mark.parametrize("clip", [False, True])
def test_prior_processing(clip, prior_data):
    """Test generating prior tensor."""
    prior = process_prior(**prior_data, clip=clip)
    if clip:
        assert (0 <= prior).all()
        assert (prior <= 1).all()
    assert prior.shape[0] == 112
    assert prior.shape[1] == 4
    assert len(prior.shape) == 2

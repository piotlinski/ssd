"""SSD data priors utils."""
from itertools import product
from math import sqrt
from typing import Iterable, List, Tuple

import torch


def unit_center(
    indices: Tuple[int, int], image_size: Tuple[int, int], stride: int
) -> Tuple[float, float]:
    """Get single prior unit center.

    :param indices: current unit's indices tuple
    :param image_size: image shape tuple
    :param stride: stride for feature map
    :return: unit center coords
    """
    y_index, x_index = indices
    y_scale, x_scale = image_size[0] / stride, image_size[1] / stride
    x_center = (x_index + 0.5) / x_scale
    y_center = (y_index + 0.5) / y_scale
    return y_center, x_center


def square_box(box_size: float, image_size: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate normalized square box shape.

    :param box_size: initial size
    :param image_size: image shape tuple
    :return: normalized square box shape
    """
    return box_size / image_size[1], box_size / image_size[0]


def rect_box(
    box_size: float, image_size: Tuple[float, float], ratio: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Calculate rectangular box shapes.

    :param box_size: initial box size
    :param image_size: image shape tuple
    :param ratio: ratio h/w for creating rectangular boxes
    :return: two normalized boxes shapes
    """
    sqrt_ratio = sqrt(ratio)
    square_width, square_height = square_box(box_size, image_size)
    return (
        (square_width * sqrt_ratio, square_height / sqrt_ratio),
        (square_width / sqrt_ratio, square_height * sqrt_ratio),
    )


def prior_boxes(
    image_size: Tuple[int, int],
    indices: Tuple[int, int],
    small_size: float,
    big_size: float,
    stride: int,
    rect_ratios: Tuple[int, ...],
) -> Iterable[Tuple[float, float, float, float]]:
    """Get prior boxes for given cell.

    :param image_size: image shape tuple
    :param indices: current unit's indices tuple
    :param small_size: small box size
    :param big_size: big box size
    :param stride: stride for the feature map
    :param rect_ratios: rectangular box ratios
    :return: Iterable of prior bounding box params
    """
    y, x = unit_center(indices, image_size, stride)

    small_square_box = (x, y, *square_box(small_size, image_size))
    yield small_square_box

    big_square_box = (x, y, *square_box(sqrt(small_size * big_size), image_size))
    yield big_square_box

    for ratio in rect_ratios:
        first, second = rect_box(small_size, image_size, ratio)
        first_rect_box = x, y, *first
        second_rect_box = x, y, *second
        yield first_rect_box
        yield second_rect_box


def feature_map_prior_boxes(
    image_size: Tuple[int, int],
    feature_map: int,
    small_size: float,
    big_size: float,
    stride: int,
    rect_ratios: Tuple[int, ...],
) -> Iterable[Tuple[float, float, float, float]]:
    """Get prior boxes for given feature map.

    :param image_size: image shape tuple
    :param feature_map: number of cells in feature map grid
    :param scale: used to normalize prior
    :param small_size: small box size
    :param big_size: big box size
    :param stride: stride for the feature map
    :param rect_ratios: rectangular box ratios
    :return: Iterable of prior bounding box params
    """
    for indices in product(range(feature_map), repeat=2):
        yield from prior_boxes(
            image_size=image_size,
            indices=indices,  # type: ignore
            small_size=small_size,
            big_size=big_size,
            stride=stride,
            rect_ratios=rect_ratios,
        )


def all_prior_boxes(
    image_size: Tuple[int, int],
    feature_maps: List[int],
    min_sizes: List[float],
    max_sizes: List[float],
    strides: List[int],
    aspect_ratios: List[Tuple[int, ...]],
) -> Iterable[Tuple[float, float, float, float]]:
    """Get prior boxes for all feature maps.

    :param image_size: size of the input image
    :param feature_maps: output channels of each backbone output
    :param min_sizes: minimal size of bbox per feature map
    :param max_sizes: maximal size of bbox per feature map
    :param strides: strides for each feature map
    :param aspect_ratios: available aspect ratios per location
        (n_boxes = 2 + ratio * 2)
    :return: Iterable of prior bounding box params
    """
    for feature_map, stride, small_size, big_size, rect_ratios in zip(
        feature_maps, strides, min_sizes, max_sizes, aspect_ratios
    ):
        yield from feature_map_prior_boxes(
            image_size=image_size,
            feature_map=feature_map,
            small_size=small_size,
            big_size=big_size,
            stride=stride,
            rect_ratios=rect_ratios,
        )


def process_prior(
    image_size: Tuple[int, int],
    feature_maps: List[int],
    min_sizes: List[float],
    max_sizes: List[float],
    strides: List[int],
    aspect_ratios: List[Tuple[int, ...]],
) -> torch.Tensor:
    """Generate SSD Prior Boxes (center, height and width of the priors)

    :param image_size: size of the input image
    :param feature_maps: output channels of each backbone output
    :param min_sizes: minimal size of bbox per feature map
    :param max_sizes: maximal size of bbox per feature map
    :param strides: strides for each feature map
    :param aspect_ratios: available aspect ratios per location
        (n_boxes = 2 + ratio * 2)
    :return: (n_priors, 4) prior boxes, relative to the image size
    """
    priors = all_prior_boxes(
        image_size=image_size,
        feature_maps=feature_maps,
        min_sizes=min_sizes,
        max_sizes=max_sizes,
        strides=strides,
        aspect_ratios=aspect_ratios,
    )
    priors_tensor = torch.tensor(data=list(priors)).clamp(min=0, max=1)
    return priors_tensor

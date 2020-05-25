"""SSD data priors utils."""
from itertools import product
from math import sqrt
from typing import Iterable, Tuple

import torch


def unit_scale(image_size: Tuple[int, int], stride: int) -> Tuple[float, float]:
    """ Calculate prior scale.

    :param image_size: image shape tuple
    :param stride: stride for feature map
    :return: scale used in feature map
    """
    return image_size[0] / stride, image_size[1] / stride


def unit_center(
    indices: Tuple[int, int], scale: Tuple[float, float]
) -> Tuple[float, float]:
    """ Get single prior unit center.

    :param indices: current unit's indices tuple
    :param scale: scale calculated using unit_scale function
    :return: unit center coords
    """
    x_index, y_index = indices
    x_scale, y_scale = scale
    x_center = (x_index + 0.5) / x_scale
    y_center = (y_index + 0.5) / y_scale
    return x_center, y_center


def square_box(box_size: float, scale: Tuple[float, float]) -> Tuple[float, float]:
    """ Calculate normalized square box shape.

    :param box_size: initial size
    :param scale: reference for normalizing
    :return: normalized square box shape
    """
    return box_size / scale[0], box_size / scale[1]


def rect_box(
    box_size: int, scale: Tuple[float, float], ratio: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """ Calculate rectangular box shapes.

    :param box_size: initial box size
    :param scale: reference for normalizing
    :param ratio: ratio h/w for creating rectangular boxes
    :return: two normalized boxes shapes
    """
    sqrt_ratio = sqrt(ratio)
    square_width, square_height = square_box(box_size, scale)
    return (
        (square_width * sqrt_ratio, square_height / sqrt_ratio),
        (square_width / sqrt_ratio, square_height * sqrt_ratio),
    )


def prior_boxes(
    indices: Tuple[int, int],
    scale: Tuple[float, float],
    small_size: int,
    big_size: int,
    rect_ratios: Tuple[int, ...],
) -> Iterable[Tuple[float, float, float, float]]:
    """ Get prior boxes for given cell.

    :param indices: current unit's indices tuple
    :param scale: used to normalize prior
    :param small_size: small box size
    :param big_size: big box size
    :param rect_ratios: rectangular box ratios
    :return: Iterable of prior bounding box params
    """
    x, y = unit_center(indices, scale)

    small_square_box = (x, y, *square_box(small_size, scale))
    yield small_square_box

    big_square_box = (x, y, *square_box(sqrt(small_size * big_size), scale))
    yield big_square_box

    for ratio in rect_ratios:
        first, second = rect_box(small_size, scale, ratio)
        first_rect_box = x, y, *first
        second_rect_box = x, y, *second
        yield first_rect_box
        yield second_rect_box


def feature_map_prior_boxes(
    feature_map: int,
    scale: Tuple[float, float],
    small_size: int,
    big_size: int,
    rect_ratios: Tuple[int, ...],
) -> Iterable[Tuple[float, float, float, float]]:
    """ Get prior boxes for given feature map.

    :param feature_map: number of cells in feature map grid
    :param scale: used to normalize prior
    :param small_size: small box size
    :param big_size: big box size
    :param rect_ratios: rectangular box ratios
    :return: Iterable of prior bounding box params
    """
    for indices in product(range(feature_map), repeat=2):
        yield from prior_boxes(
            indices=indices,  # type: ignore
            scale=scale,
            small_size=small_size,
            big_size=big_size,
            rect_ratios=rect_ratios,
        )


def all_prior_boxes(
    image_size: Tuple[int, int],
    feature_maps: Tuple[int, ...],
    min_sizes: Tuple[int, ...],
    max_sizes: Tuple[int, ...],
    strides: Tuple[int, ...],
    aspect_ratios: Tuple[Tuple[int, ...], ...],
) -> Iterable[Tuple[float, float, float, float]]:
    """ Get prior boxes for all feature maps.

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
        scale = unit_scale(image_size, stride)
        yield from feature_map_prior_boxes(
            feature_map=feature_map,
            scale=scale,
            small_size=small_size,
            big_size=big_size,
            rect_ratios=rect_ratios,
        )


def process_prior(
    image_size: Tuple[int, int],
    feature_maps: Tuple[int, ...],
    min_sizes: Tuple[int, ...],
    max_sizes: Tuple[int, ...],
    strides: Tuple[int, ...],
    aspect_ratios: Tuple[Tuple[int, ...], ...],
    clip: bool,
):
    """ Generate SSD Prior Boxes (center, height and width of the priors)

    :param image_size: size of the input image
    :param feature_maps: output channels of each backbone output
    :param min_sizes: minimal size of bbox per feature map
    :param max_sizes: maximal size of bbox per feature map
    :param strides: strides for each feature map
    :param aspect_ratios: available aspect ratios per location
        (n_boxes = 2 + ratio * 2)
    :param clip: clip params to [0, 1]
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
    priors_tensor = torch.tensor(data=list(priors))
    if clip:
        priors_tensor.clamp_(min=0, max=1)
    return priors_tensor

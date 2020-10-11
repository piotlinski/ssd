"""SSD bounding box utils."""
from typing import Tuple

import torch


def convert_locations_to_boxes(
    locations: torch.Tensor,
    priors: torch.Tensor,
    center_variance: float,
    size_variance: float,
) -> torch.Tensor:
    """Convert regressional location results of SSD into boxes in the form of
        (center_x, center_y, h, w)

    $$hat{center} * center_variance = \frac {center - center_prior} {hw_prior}$$
    $$exp(hat{hw} * size_variance) = \frac {hw} {hw_prior}$$

    :param locations: (batch_size, num_priors, 4) the regression output of SSD,
        containing the outputs
    :param priors: (num_priors, 4) or (batch_size/1, num_priors, 4) prior boxes
    :param center_variance: changes the scale of center
    :param size_variance: changes scale of size
    :return: priors [[center_x, center_y, w, h]], relative to the image size
    """
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)

    centers = locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2]
    hws = torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]

    return torch.cat([centers, hws], dim=locations.dim() - 1)


def convert_boxes_to_locations(
    boxes: torch.Tensor,
    priors: torch.Tensor,
    center_variance: float,
    size_variance: float,
) -> torch.Tensor:
    """Convert boxes (x, y, w, h) to regressional location results of SSD

    $$hat{center} * center_variance = \frac {center - center_prior} {hw_prior}$$
    $$exp(hat{hw} * size_variance) = \frac {hw} {hw_prior}$$

    :param boxes: center form boxes
    :param priors: center form priors
    :param center_variance: changes the scale of center
    :param size_variance: changes scale of size
    :return: locations for training SSD
    """
    if priors.dim() + 1 == boxes.dim():
        priors = priors.unsqueeze(0)
    centers = (boxes[..., :2] - priors[..., :2]) / priors[..., 2:] / center_variance
    hws = torch.log(boxes[..., 2:] / priors[..., 2:]) / size_variance
    return torch.cat([centers, hws], dim=boxes.dim() - 1)


def center_bbox_to_corner_bbox(center_bboxes: torch.Tensor) -> torch.Tensor:
    """Convert x, y, w, h form to x1, y1, x2, y2."""
    point_1 = center_bboxes[..., :2] - center_bboxes[..., 2:] / 2
    point_2 = center_bboxes[..., :2] + center_bboxes[..., 2:] / 2
    return torch.cat([point_1, point_2], center_bboxes.dim() - 1)


def corner_bbox_to_center_bbox(corner_bboxes: torch.Tensor) -> torch.Tensor:
    """Convert x1, y1, x2, y2 form to x, y, w, h."""
    xy = (corner_bboxes[..., :2] + corner_bboxes[..., 2:]) / 2
    wh = corner_bboxes[..., 2:] - corner_bboxes[..., :2]
    return torch.cat([xy, wh], corner_bboxes.dim() - 1)


def area(left_top: torch.Tensor, right_bottom: torch.Tensor):
    """Compute area of rectangles given two corners.

    :param left_top: (N, 2) left top corner
    :param right_bottom: (N, 2) right bottom corner
    :return: (N) area of the rectangle
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
    """Return intersection-over-union (Jaccard index) of boxes.

    :param boxes_1: (N, 4) ground truth boxes
    :param boxes_2: (N or 1, 4) predicted boxes
    :return: (N) IoU values
    """
    overlap_ltop = torch.max(boxes_1[..., :2], boxes_2[..., :2])
    overlap_rbot = torch.min(boxes_1[..., 2:], boxes_2[..., 2:])
    intersection = area(overlap_ltop, overlap_rbot)

    boxes_1_area = area(boxes_1[..., :2], boxes_1[..., 2:])
    boxes_2_area = area(boxes_2[..., :2], boxes_2[..., 2:])
    union = boxes_1_area + boxes_2_area - intersection

    return intersection / (union + 1e-5)


def assign_priors(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    corner_form_priors: torch.Tensor,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign ground truth boxes and targets to priors.

    :param gt_boxes: (num_targets, 4) ground truth boxes
    :param gt_labels: (num_targets) labels of targets
    :param corner_form_priors: (num_priors, 4) corner form priors
    :param iou_threshold: IoU threshold for assigning
    :return: boxes (num_priors, 4) and labels (num_priors)
    """
    ious = iou(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    best_target_per_prior, tpp_idx = ious.max(1)
    best_prior_per_target, ppt_idx = ious.max(0)

    for target_idx, prior_idx in enumerate(ppt_idx):
        tpp_idx[prior_idx] = target_idx

    best_target_per_prior.index_fill_(0, ppt_idx, 2)  # 2 -> every target has a prior
    labels = gt_labels[tpp_idx]
    labels[best_target_per_prior < iou_threshold] = 0  # background class
    boxes = gt_boxes[tpp_idx]
    return boxes, labels

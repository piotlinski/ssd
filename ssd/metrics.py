"""Object detection metrics."""
from typing import Tuple

import torch
from torchnet.meter import mAPMeter

from ssd.data.bboxes import iou
from ssd.data.datasets import onehot_labels


def assign_predictions(
    gt_boxes: torch.Tensor, pred_boxes: torch.Tensor
) -> torch.Tensor:
    """ Assign predicted boxes to ground truth.

    :param gt_boxes: ground truth boxes coordinates (N x 4): x1, y1, x2, y2
    :param pred_boxes: predicted boxes coordinates (M x 4): x1, y1, x2, y2
    :return: indices of gt to which prediction was assigned (M x 2): IOU, gt index
    """
    return torch.tensor(
        [
            torch.max(iou(gt_boxes, pred_box.unsqueeze(0)), dim=0)
            for pred_box in pred_boxes
        ]
    )


def sort_by_confidence(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort prediction by confidence."""
    sorted_scores, indices = torch.sort(scores, descending=True)
    return boxes[indices], sorted_scores, labels[indices]


def adjust_labels(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Modify incorrect labels to an incorrect label value to fit mAP definition
    (ensure it is a FP)

    :param gt_boxes: ground truth boxes coordinates (N x 4): x1, y1, x2, y2
    :param gt_labels: ground truth labels (N)
    :param pred_boxes: predicted boxes coordinates (M x 4): x1, y1, x2, y2
    :param pred_labels: predicted labels (M)
    :param iou_threshold: intersection over union threshold to set box as TP
    :return: tensors of output and target (M)
    """
    wrong_class = max(max(gt_labels), max(pred_labels)) + 1
    pred_assignment = assign_predictions(gt_boxes=gt_boxes, pred_boxes=pred_boxes)
    target_labels = gt_labels[pred_assignment[:, 1].long()]
    output_labels = pred_labels.clone()
    gt_assigned = [False] * gt_boxes.shape[0]
    for idx, (pred_iou, pred_gt_idx) in enumerate(pred_assignment):
        gt_idx = int(pred_gt_idx)
        if pred_iou < iou_threshold or gt_assigned[gt_idx]:
            output_labels[idx] = wrong_class
        else:
            gt_assigned[gt_idx] = True
    return output_labels, target_labels


def mean_average_precision(
    gt_boxes_batch: torch.Tensor,
    gt_labels_batch: torch.Tensor,
    pred_boxes_batch: torch.Tensor,
    pred_scores_batch: torch.Tensor,
    pred_labels_batch: torch.Tensor,
    iou_threshold: float = 0.5,
) -> float:
    """ Calculate mean average precision for given output and target.

    :param gt_boxes_batch: ground truth boxes coordinates batch (b x N x 4)
    :param gt_labels_batch: ground truth labels batch (b x N)
    :param pred_boxes_batch: predicted boxes coordinates batch (b x M x 4)
    :param pred_scores_batch: predicted scores batch (b x M)
    :param pred_labels_batch: predicted labels batch (b x M)
    :param iou_threshold: intersection over union threshold to set box as TP
    :return: mAP value
    """
    meter = mAPMeter()
    for gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels in zip(
        gt_boxes_batch,
        gt_labels_batch,
        pred_boxes_batch,
        pred_scores_batch,
        pred_labels_batch,
    ):
        pred_boxes_sorted, pred_scores_sorted, pred_labels_sorted = sort_by_confidence(
            boxes=pred_boxes, scores=pred_scores, labels=pred_labels
        )
        output_labels, target_labels = adjust_labels(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes_sorted,
            pred_labels=pred_labels_sorted,
            iou_threshold=iou_threshold,
        )
        n_classes = max(max(output_labels), max(target_labels)) + 1
        output = onehot_labels(
            labels=output_labels, n_classes=n_classes
        ) * pred_scores_sorted.unsqueeze(1)
        target = onehot_labels(labels=target_labels, n_classes=n_classes)
        meter.add(output=output, target=target)
    return meter.value()

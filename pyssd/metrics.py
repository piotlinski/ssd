"""Object detection metrics."""
import warnings
from typing import Iterable, Tuple

import torch
from torchnet.meter import mAPMeter

from pyssd.data.bboxes import iou
from pyssd.data.datasets import onehot_labels


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
    wrong_class: int,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Modify incorrect labels to an incorrect label value to fit mAP definition
    (ensure it is a FP)

    :param gt_boxes: ground truth boxes coordinates (N x 4): x1, y1, x2, y2
    :param gt_labels: ground truth labels (N)
    :param pred_boxes: predicted boxes coordinates (M x 4): x1, y1, x2, y2
    :param pred_labels: predicted labels (M)
    :param wrong_class: id of an incorrect class label
    :param iou_threshold: intersection over union threshold to set box as TP
    :return: tensors of output and target (M)
    """
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
    gt_boxes_batch: Iterable[torch.Tensor],
    gt_labels_batch: Iterable[torch.Tensor],
    pred_boxes_batch: Iterable[torch.Tensor],
    pred_scores_batch: Iterable[torch.Tensor],
    pred_labels_batch: Iterable[torch.Tensor],
    n_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    """ Calculate mean average precision for given output and target.

    :param gt_boxes_batch: ground truth boxes coordinates batch (b x N x 4)
    :param gt_labels_batch: ground truth labels batch (b x N)
    :param pred_boxes_batch: predicted boxes coordinates batch (b x M x 4)
    :param pred_scores_batch: predicted scores batch (b x M)
    :param pred_labels_batch: predicted labels batch (b x M)
    :param n_classes: number of classes
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
        if gt_labels.numel() == 0 or pred_labels.numel() == 0:
            meter.add(
                output=torch.zeros(1, n_classes + 1),
                target=torch.zeros(1, n_classes + 1),
            )
            continue
        pred_boxes_sorted, pred_scores_sorted, pred_labels_sorted = sort_by_confidence(
            boxes=pred_boxes, scores=pred_scores, labels=pred_labels
        )
        output_labels, target_labels = adjust_labels(
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            pred_boxes=pred_boxes_sorted,
            pred_labels=pred_labels_sorted,
            wrong_class=n_classes,
            iou_threshold=iou_threshold,
        )
        output = onehot_labels(
            labels=output_labels, n_classes=n_classes + 1
        ) * pred_scores_sorted.unsqueeze(1)
        target = onehot_labels(labels=target_labels, n_classes=n_classes + 1)
        meter.add(output=output, target=target)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="indexing with dtype torch.uint8 is now deprecated,"
            " please use a dtype torch.bool instead",
        )
        m_ap = meter.value()
    return m_ap

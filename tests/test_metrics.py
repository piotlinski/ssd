"""Test object detection metrics."""
import pytest
import torch

from ssd.metrics import (
    adjust_labels,
    assign_predictions,
    mean_average_precision,
    sort_by_confidence,
)


@pytest.fixture
def example_gt_and_prediction():
    """Example GT and prediction boxes and labels."""
    gt_boxes = torch.tensor([[0, 0, 5, 5], [10, 10, 15, 15], [20, 20, 25, 25]])
    gt_labels = torch.tensor([3, 2, 1])
    pred_boxes = torch.tensor(
        [
            [20, 20, 25, 29],  # fits last with correct label (depends on IOU threshold)
            [1, 0, 6, 5],  # fits first with wrong label (False)
            [10, 10, 15, 15],  # fits second ideally (True)
            [10, 10, 15, 15],  # fits second ideally (False - already assigned)
            [50, 51, 57, 59],  # does not fit (False)
        ]
    )
    pred_scores = torch.tensor([0.2, 0.8, 0.4, 0.3, 0.5])
    pred_labels = torch.tensor([1, 2, 2, 2, 3])
    return gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels


def test_assign_predictions(example_gt_and_prediction):
    """Verify if predictions are assigned to ground truth correctly."""
    gt_boxes, _, pred_boxes, *_ = example_gt_and_prediction
    assignment = assign_predictions(gt_boxes, pred_boxes)
    assert assignment[0, 1] == 2
    assert assignment[1, 1] == 0
    assert assignment[2, 1] == 1
    assert assignment[3, 1] == 1
    assert assignment.shape[0] == 5


def test_sort_by_confidence(example_gt_and_prediction):
    """Verify if tensors are sorted by confidence tensor."""
    *_, boxes, scores, labels = example_gt_and_prediction
    s_boxes, s_scores, s_labels = sort_by_confidence(boxes, scores, labels)
    assert (
        s_boxes
        == torch.tensor(
            [
                [1, 0, 6, 5],
                [50, 51, 57, 59],
                [10, 10, 15, 15],
                [10, 10, 15, 15],
                [20, 20, 25, 29],
            ]
        )
    ).all()
    assert (s_scores == torch.tensor([0.8, 0.5, 0.4, 0.3, 0.2])).all()
    assert (s_labels == torch.tensor([2, 3, 2, 2, 1])).all()


@pytest.mark.parametrize("iou_threshold", [0.5, 0.75])
def test_adjust_labels(iou_threshold, example_gt_and_prediction):
    """Verify modifying labels for TP and FP."""
    (gt_boxes, gt_labels, pred_boxes, _, pred_labels,) = example_gt_and_prediction
    output_labels, target_labels = adjust_labels(
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        iou_threshold=iou_threshold,
    )
    assert (output_labels[0] == target_labels[0]) == (iou_threshold == 0.5)
    assert output_labels[1] != target_labels[1]
    assert output_labels[2] == target_labels[2]
    assert output_labels[3] != target_labels[3]
    assert output_labels[3] == 4
    assert output_labels[4] != target_labels[4]
    assert output_labels[4] == 4


def test_mean_average_precision(example_gt_and_prediction):
    """Verify calculating mAP in batch."""
    (
        gt_boxes,
        gt_labels,
        pred_boxes,
        pred_scores,
        pred_labels,
    ) = example_gt_and_prediction
    data = dict(
        gt_boxes_batch=gt_boxes.unsqueeze(0),
        gt_labels_batch=gt_labels.unsqueeze(0),
        pred_boxes_batch=pred_boxes.unsqueeze(0),
        pred_scores_batch=pred_scores.unsqueeze(0),
        pred_labels_batch=pred_labels.unsqueeze(0),
    )
    mean_ap_50 = mean_average_precision(**data, iou_threshold=0.5)
    mean_ap_75 = mean_average_precision(**data, iou_threshold=0.75)
    mean_ap_90 = mean_average_precision(**data, iou_threshold=0.9)
    mean_ap_99 = mean_average_precision(**data, iou_threshold=0.99)
    assert mean_ap_99 <= mean_ap_90 <= mean_ap_75 <= mean_ap_50

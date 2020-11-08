"""Visualization utils."""
from typing import Any, Dict, Iterable, List

import torch


def create_box_list(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_labels: List[str],
    label_prefix: str = "",
) -> Iterable[Dict[str, Any]]:
    """Create list with box information for wandb."""
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int()
        yield {
            "position": {
                "minX": x1.item(),
                "maxX": x2.item(),
                "minY": y1.item(),
                "maxY": y2.item(),
            },
            "class_id": label.item(),
            "box_caption": f"{label_prefix}{class_labels[int(label.item() - 1)]}",
            "domain": "pixel",
            "scores": {"score": score.item()},
        }


def get_boxes(
    gt_boxes: torch.Tensor,
    gt_scores: torch.Tensor,
    gt_labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_labels: List[str],
):
    """Get wandb image with predictions."""
    boxes = {
        "predictions": {
            "box_data": list(
                create_box_list(
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    class_labels=class_labels,
                )
            ),
            "class_labels": dict(enumerate(class_labels, start=1)),
        },
        "ground_truth": {
            "box_data": list(
                create_box_list(
                    boxes=gt_boxes,
                    scores=gt_scores,
                    labels=gt_labels,
                    class_labels=class_labels,
                    label_prefix="gt_",
                )
            ),
            "class_labels": dict(enumerate(class_labels, start=1)),
        },
    }
    return boxes

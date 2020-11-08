"""Test visualization tools."""
import torch

from pyssd.modeling.visualize import get_boxes


def test_get_boxes():
    """Verify format of boxes fetched from get_boxes."""
    gt_boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8]])
    gt_scores = torch.tensor([1.0, 1.0])
    gt_labels = torch.tensor([1.0, 2.0])
    boxes = torch.tensor([[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])
    scores = torch.tensor([0.25, 0.75])
    labels = torch.tensor([2.0, 1.0])
    class_labels = ["dog", "cat"]

    boxes_dict = get_boxes(
        gt_boxes=gt_boxes,
        gt_scores=gt_scores,
        gt_labels=gt_labels,
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_labels=class_labels,
    )
    assert boxes_dict == {
        "predictions": {
            "box_data": [
                {
                    "position": {
                        "minX": 2,
                        "maxX": 4,
                        "minY": 3,
                        "maxY": 5,
                    },
                    "class_id": 2.0,
                    "box_caption": "cat",
                    "domain": "pixel",
                    "scores": {"score": 0.25},
                },
                {
                    "position": {
                        "minX": 6,
                        "maxX": 8,
                        "minY": 7,
                        "maxY": 9,
                    },
                    "class_id": 1.0,
                    "box_caption": "dog",
                    "domain": "pixel",
                    "scores": {"score": 0.75},
                },
            ],
            "class_labels": {1: "dog", 2: "cat"},
        },
        "ground_truth": {
            "box_data": [
                {
                    "position": {
                        "minX": 1,
                        "maxX": 3,
                        "minY": 2,
                        "maxY": 4,
                    },
                    "class_id": 1.0,
                    "box_caption": "gt_dog",
                    "domain": "pixel",
                    "scores": {"score": 1.0},
                },
                {
                    "position": {
                        "minX": 5,
                        "maxX": 7,
                        "minY": 6,
                        "maxY": 8,
                    },
                    "class_id": 2.0,
                    "box_caption": "gt_cat",
                    "domain": "pixel",
                    "scores": {"score": 1.0},
                },
            ],
            "class_labels": {1: "dog", 2: "cat"},
        },
    }

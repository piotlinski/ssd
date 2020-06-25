"""Visualization utils."""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib import patches
from yacs.config import CfgNode

from ssd.modeling.model import process_model_prediction


def plot_image(
    config: CfgNode,
    image: torch.tensor,
    prediction: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ground_truth: bool = False,
) -> plt.Axes:
    """ Plot an image with predicted bounding boxes.

    :param config: SSD config
    :param image: Input image
    :param prediction: optional class logits and bbox predictions for the image
        (keep batch dimension as 1)
    :param ground_truth: plotting ground truth (this modifies confidence threshold)
    :return: matplotlib axis with image and optional bounding boxes
    """
    fig, ax = plt.subplots(frameon=False)
    fig.tight_layout()
    ax.axis("off")
    numpy_image = image.squeeze(0).numpy()
    ax.imshow(numpy_image, cmap="gray")
    if prediction is not None:
        colors = plt.cm.get_cmap("Dark2")
        cls_logits, bbox_pred = prediction
        plot_config = config.clone()
        if ground_truth:
            plot_config.defrost()
            plot_config.MODEL.CONFIDENCE_THRESHOLD = 0.1  # to filter repeated boxes
        ((boxes, scores, labels),) = process_model_prediction(
            plot_config, cls_logits, bbox_pred
        )
        for box, score, label in zip(boxes, scores, labels):
            color = colors(label.item() / (config.DATA.N_CLASSES - 1))
            x1, y1, x2, y2 = box.int()
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
            )
            ax.text(
                x1,
                y2,
                f"{label.item() - 1:.0f}: {score.item():.2f}",
                verticalalignment="top",
                color="w",
                fontsize="x-small",
                fontweight="semibold",
                clip_on=True,
                bbox=dict(pad=0, facecolor=color, alpha=0.8),
            )
            ax.add_patch(rect)
    return ax

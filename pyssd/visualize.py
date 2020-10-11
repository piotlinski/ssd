"""Visualization utils."""
from random import shuffle
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib import patches
from yacs.config import CfgNode

from pyssd.modeling.model import process_model_prediction


def plot_image(
    config: CfgNode,
    image: torch.tensor,
    prediction: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ax: Optional[plt.Axes] = None,
    confidence_threshold: Optional[float] = None,
) -> plt.Axes:
    """Plot an image with predicted bounding boxes.

    :param config: SSD config
    :param image: Input image
    :param prediction: optional class logits and bbox predictions for the image
        (keep batch dimension as 1)
    :param ax: optional axis to plot on
    :param confidence_threshold: Optional confidence threshold to set
    :return: matplotlib axis with image and optional bounding boxes
    """
    if ax is None:
        ax = plt.gca()
    ax.axis("off")
    label_names = config.DATA.CLASS_LABELS
    numpy_image = image.cpu().numpy()
    ax.imshow(numpy_image)
    if prediction is not None:
        colors = plt.cm.get_cmap("Dark2")
        cls_logits, bbox_pred = prediction
        plot_config = config.clone()
        if confidence_threshold is not None:
            plot_config.defrost()
            plot_config.MODEL.CONFIDENCE_THRESHOLD = confidence_threshold
        ((boxes, scores, labels),) = process_model_prediction(
            plot_config, cls_logits.unsqueeze(0), bbox_pred.unsqueeze(0)
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
                f"{label_names[int(label.item() - 1)]}: {score.item():.2f}",
                verticalalignment="top",
                color="w",
                fontsize="x-small",
                fontweight="semibold",
                clip_on=True,
                bbox=dict(pad=0, facecolor=color, alpha=0.8),
            )
            ax.add_patch(rect)
    return ax


def plot_images_from_batch(
    config: CfgNode,
    image_batch: torch.Tensor,
    pred_cls_logits: torch.Tensor,
    pred_bbox_pred: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_bbox_pred: torch.Tensor,
) -> plt.Figure:
    """Randomly select images from batch and plot with varying confidence.

    :param config: SSD config
    :param image_batch: image batch
    :param pred_cls_logits: predicted cls_logits
    :param pred_bbox_pred: predicted bbox_pred
    :param gt_labels: ground truth labels
    :param gt_bbox_pred: ground truth bbox_pred
    :return: figure with visualization
    """
    n_examples = config.RUNNER.VIS_N_IMAGES
    indices = list(range(image_batch.shape[0]))
    shuffle(indices)
    confidence_thresholds = config.RUNNER.VIS_CONFIDENCE_THRESHOLDS
    fig = plt.Figure(figsize=(4 * (len(confidence_thresholds) + 1), 4 * n_examples))
    for idx, example_idx in enumerate(indices[:n_examples]):
        image = image_batch[example_idx].permute(1, 2, 0)
        denominator = torch.reciprocal(
            torch.tensor(config.DATA.PIXEL_STD, device=image.device)
        )
        image = image / denominator + torch.tensor(
            config.DATA.PIXEL_MEAN, device=image.device
        )
        image.clamp_(min=0, max=1)
        subplot_idx = idx * (len(confidence_thresholds) + 1) + 1
        ax = fig.add_subplot(n_examples, len(confidence_thresholds) + 1, subplot_idx)
        plot_image(
            config,
            image=image,
            prediction=(
                gt_labels[example_idx],
                gt_bbox_pred[example_idx],
            ),
            ax=ax,
        )
        ax.set_title("gt")
        for conf_idx, conf in enumerate(confidence_thresholds, start=1):
            ax = fig.add_subplot(
                n_examples, len(confidence_thresholds) + 1, subplot_idx + conf_idx
            )
            plot_image(
                config,
                image=image,
                prediction=(
                    pred_cls_logits[example_idx],
                    pred_bbox_pred[example_idx],
                ),
                ax=ax,
                confidence_threshold=conf,
            )
            ax.set_title(f"thresh={conf}")
    return fig

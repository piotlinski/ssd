"""SSD model."""
import logging
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms
from yacs.config import CfgNode

from pyssd.data.bboxes import center_bbox_to_corner_bbox, convert_locations_to_boxes
from pyssd.data.datasets import onehot_labels
from pyssd.data.priors import process_prior
from pyssd.modeling.backbones import backbones
from pyssd.modeling.box_predictors import box_predictors

logger = logging.getLogger(__name__)


class SSD(nn.Module):
    """SSD Detector class."""

    def __init__(self, config: CfgNode):
        super().__init__()
        backbone = backbones[config.MODEL.BACKBONE]
        self.backbone = backbone(config)
        predictor = box_predictors[config.MODEL.BOX_PREDICTOR]
        self.predictor = predictor(
            config,
            backbone_out_channels=self.backbone.out_channels,
        )

    def forward(self, images):
        features = self.backbone(images)
        predictions = self.predictor(features)
        return predictions


def process_model_output(
    detections: Tuple[torch.Tensor, torch.Tensor],
    image_size: Tuple[int, int],
    confidence_threshold: float,
    nms_threshold: float,
    max_per_image: int,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Process model output with non-max suppression.

    :param detections: tuple of class logits and bounding box regressions
    :param image_size: input image shape tuple
    :param confidence_threshold: min confidence to use prediction
    :param nms_threshold: non-maximum suppresion threshold
    :param max_per_image: max number of detections per image
    :return: iterable of tuples containing bounding boxes, scores and labels
    """
    width, height = image_size
    scores_batches, boxes_batches = detections
    batch_size = scores_batches.size(0)
    device = scores_batches.device
    for batch_idx in range(batch_size):
        scores = scores_batches[batch_idx]  # (N, num_classes)
        boxes = boxes_batches[batch_idx]  # (N, 4)
        n_boxes, n_classes = scores.shape

        boxes = boxes.view(n_boxes, 1, 4).expand(n_boxes, n_classes, 4)
        labels = torch.arange(n_classes, device=device)
        labels = labels.view(1, n_classes).expand_as(scores)

        # remove predictions with label == background
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        approved_mask = (
            (scores > confidence_threshold).nonzero(as_tuple=False).squeeze(1)
        )
        boxes = boxes[approved_mask]
        scores = scores[approved_mask]
        labels = labels[approved_mask]

        # reshape boxes to image size
        boxes[:, 0::2] *= width
        boxes[:, 1::2] *= height

        # as of torchvision 0.6.0, cuda nms is broken (int overflow)
        try:
            keep_mask = batched_nms(
                boxes=boxes,
                scores=scores,
                idxs=labels,
                iou_threshold=nms_threshold,
            )
        except RuntimeError:
            logger.warning("Torchvision NMS CUDA int overflow. Falling back to CPU.")
            keep_mask = batched_nms(
                boxes=boxes.cpu(),
                scores=scores.cpu(),
                idxs=labels.cpu(),
                iou_threshold=nms_threshold,
            )
        # keep only top scoring predictions
        keep_mask = keep_mask[:max_per_image]
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

        yield boxes, scores, labels


def process_model_prediction(
    config: CfgNode, cls_logits: torch.Tensor, bbox_pred: torch.Tensor
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Get readable results from model predictions.

    :param config: SSD configuration
    :param cls_logits: class predictions from model
    :param bbox_pred: bounding box predictions from model
    :return: list of predictions - tuples of boxes, scores and labels
    """
    priors = process_prior(
        image_size=config.DATA.SHAPE,
        feature_maps=config.DATA.PRIOR.FEATURE_MAPS,
        min_sizes=config.DATA.PRIOR.MIN_SIZES,
        max_sizes=config.DATA.PRIOR.MAX_SIZES,
        strides=config.DATA.PRIOR.STRIDES,
        aspect_ratios=config.DATA.PRIOR.ASPECT_RATIOS,
        clip=config.DATA.PRIOR.CLIP,
    ).to(cls_logits.device)
    if len(cls_logits.shape) == 3:
        scores = nn.functional.softmax(cls_logits, dim=2)
    else:
        scores = onehot_labels(labels=cls_logits, n_classes=config.DATA.N_CLASSES)
    boxes = convert_locations_to_boxes(
        locations=bbox_pred,
        priors=priors,
        center_variance=config.MODEL.CENTER_VARIANCE,
        size_variance=config.MODEL.SIZE_VARIANCE,
    )
    boxes = center_bbox_to_corner_bbox(boxes)
    detections = process_model_output(
        detections=(scores, boxes),
        image_size=config.DATA.SHAPE,
        confidence_threshold=config.MODEL.CONFIDENCE_THRESHOLD,
        nms_threshold=config.MODEL.NMS_THRESHOLD,
        max_per_image=config.MODEL.MAX_PER_IMAGE,
    )
    return list(detections)

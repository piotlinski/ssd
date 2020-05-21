"""SSD model."""
import torch
import torch.nn as nn
from torchvision.ops import nms
from yacs.config import CfgNode

from ssd.modeling.backbones import backbones
from ssd.modeling.box_predictors import box_predictors


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
            boxes_per_location=config.DATA.PRIOR.BOXES_PER_LOC,
        )

    def forward(self, images):
        features = self.backbone(images)
        predictions = self.predictor(features)
        return predictions


def non_max_suppression(
    boxes: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor, threshold: float
) -> torch.Tensor:
    """ Perform non-maximum suppression.

    ..  strategy: in order to perform NMS independently per class. We add an offset to
        all the boxes. The offset is dependent only on the class idx, and is large
        enough so that boxes from different classes do not overlap

    :param boxes: (N, 4) boxes where NMS will be performed. They are expected to be in
        (x1, y1, x2, y2) format
    :param scores: (N) scores for each one of the boxes
    :param indices: (N) indices of the categories for each one of the boxes
    :param threshold: IoU threshold for discarding overlapping boxes
    :return: mask of indices kept
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coord = boxes.max()
    offsets = indices.to(boxes) * (max_coord + 1)
    nms_boxes = boxes + offsets[:, None]
    return nms(boxes=nms_boxes, scores=scores, iou_threshold=threshold)

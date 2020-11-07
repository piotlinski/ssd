"""SSD model."""
import logging
from typing import Any, Iterable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.ops.boxes import batched_nms

from pyssd.data.bboxes import center_bbox_to_corner_bbox, convert_locations_to_boxes
from pyssd.data.datasets import onehot_labels
from pyssd.data.priors import process_prior
from pyssd.data.transforms import SSDTargetTransform
from pyssd.loss import MultiBoxLoss
from pyssd.modeling.backbones import backbones
from pyssd.modeling.box_predictors import box_predictors

logger = logging.getLogger(__name__)


class SSD(pl.LightningModule):
    """SSD Detector class."""

    def __init__(
        self,
        n_classes: int,
        class_labels: List[str],
        object_label: str,
        lr: float = 1e-3,
        backbone_name: str = "VGG300",
        use_pretrained_backbone: bool = False,
        predictor_name: str = "SSD",
        image_size: Tuple[int, int] = (300, 300),
        center_variance: float = 0.1,
        size_variance: float = 0.2,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.2,
        nms_threshold: float = 0.45,
        max_per_image: int = 100,
        negative_positive_ratio: float = 3,
    ):
        """
        :param n_classes: number of classes (if 2 then no classification)
        :param class_labels: dataset class labels
        :param object_label: dataset object label
        :param lr: learning rate
        :param backbone_name: used backbone name
        :param use_pretrained_backbone: download pretrained weights for backbone
        :param predictor_name: used predictor name
        :param image_size: image size tuple
        :param center_variance: SSD center variance
        :param size_variance: SSD size variance
        :param iou_threshold: IOU threshold for anchors
        :param confidence_threshold: min prediction confidence to use as detection
        :param nms_threshold: non-max suppression IOU threshold
        :param max_per_image: max number of detections per image
        :param negative_positive_ratio: the ratio between the negative examples and
            positive examples for calculating loss
        """
        super().__init__()
        backbone = backbones[backbone_name]
        self.backbone = backbone(use_pretrained=use_pretrained_backbone)
        predictor = box_predictors[predictor_name]
        self.predictor = predictor(
            n_classes=n_classes,
            backbone_out_channels=self.backbone.out_channels,
            backbone_boxes_per_loc=self.backbone.boxes_per_loc,
        )
        self.anchors = nn.Parameter(
            process_prior(
                image_size=image_size,
                feature_maps=self.backbone.feature_maps,
                min_sizes=self.backbone.min_sizes,
                max_sizes=self.backbone.max_sizes,
                strides=self.backbone.strides,
                aspect_ratios=self.backbone.aspect_ratios,
            ),
            requires_grad=False,
        )
        self.target_transform = SSDTargetTransform(
            anchors=self.anchors,
            image_size=image_size,
            n_classes=n_classes,
            center_variance=center_variance,
            size_variance=size_variance,
            iou_threshold=iou_threshold,
        )
        self.image_size = image_size
        self.n_classes = n_classes
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_per_image = max_per_image
        self.class_labels = class_labels if n_classes != 2 else [object_label]

        self.criterion = MultiBoxLoss(negative_positive_ratio)
        self.lr = lr

    def process_model_output(
        self, detections: Tuple[torch.Tensor, torch.Tensor], confidence_threshold: float
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Process model output with non-max suppression.

        :param detections: tuple of class logits and bounding box regressions
        :param confidence_threshold: min detection confidence threshold
        :return: iterable of tuples containing bounding boxes, scores and labels
        """
        width, height = self.image_size
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
                    iou_threshold=self.nms_threshold,
                )
            except RuntimeError:
                logger.warning(
                    "Torchvision NMS CUDA int overflow. Falling back to CPU."
                )
                keep_mask = batched_nms(
                    boxes=boxes.cpu(),
                    scores=scores.cpu(),
                    idxs=labels.cpu(),
                    iou_threshold=self.nms_threshold,
                )
            # keep only top scoring predictions
            keep_mask = keep_mask[: self.max_per_image]
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            labels = labels[keep_mask]

            yield boxes, scores, labels

    def process_model_prediction(
        self,
        cls_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        confidence_threshold: Optional[float] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get readable results from model predictions.

        :param cls_logits: class predictions from model
        :param bbox_pred: bounding box predictions from model
        :param confidence_threshold: optional param to override default threshold
        :return: list of predictions - tuples of boxes, scores and labels
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        if len(cls_logits.shape) == 3:
            scores = functional.softmax(cls_logits, dim=2)
        else:
            scores = onehot_labels(labels=cls_logits, n_classes=self.n_classes)
        boxes = convert_locations_to_boxes(
            locations=bbox_pred,
            priors=self.anchors,
            center_variance=self.center_variance,
            size_variance=self.size_variance,
        )
        boxes = center_bbox_to_corner_bbox(boxes)
        detections = self.process_model_output(
            detections=(scores, boxes), confidence_threshold=confidence_threshold
        )
        return list(detections)

    def forward(
        self, images: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward function for inference."""
        features = self.backbone(images)
        cls_logits, bbox_pred = self.predictor(features)
        return self.process_model_prediction(cls_logits, bbox_pred)

    def training_step(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Step for training."""
        images, locations, labels = data

        features = self.backbone(images)
        cls_logits, bbox_pred = self.predictor(features)

        regression_loss, classification_loss = self.criterion(
            confidence=cls_logits,
            predicted_locations=bbox_pred,
            labels=labels,
            gt_locations=locations,
        )
        loss = regression_loss + classification_loss

        return {
            "loss": loss,
            "log": {
                "loss-regression/train": regression_loss,
                "loss-classification/train": classification_loss,
                "loss-total/train": loss,
            },
        }

    def validation_step(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Step for validation."""
        return self.training_step(data)

    def validation_epoch_end(self, outputs: List[Any]):
        """Summarize after validation."""
        regression_loss = []
        classification_loss = []
        loss = []
        for output in outputs:
            regression_loss.append(output["log"]["loss-regression/train"])
            classification_loss.append(output["log"]["loss-classification/train"])
            loss.append(output["loss"])

        return {
            "val_loss": torch.tensor(loss).mean(),
            "log": {
                "loss-regression/val": torch.tensor(regression_loss).mean(),
                "loss-classification/val": torch.tensor(classification_loss).mean(),
                "loss-total/val": torch.tensor(loss).mean(),
            },
        }

    def configure_optimizers(self):
        """Configure training optimizer."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer, patience=self.lr_reduce_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

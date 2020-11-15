"""SSD model."""
import logging
from argparse import ArgumentParser
from typing import Iterable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as functional
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data.dataloader import DataLoader
from torchvision.ops.boxes import batched_nms

from pytorch_ssd.data.bboxes import (
    center_bbox_to_corner_bbox,
    convert_locations_to_boxes,
)
from pytorch_ssd.data.datasets import datasets, onehot_labels
from pytorch_ssd.data.priors import process_prior
from pytorch_ssd.data.transforms import (
    DataTransform,
    SSDTargetTransform,
    TrainDataTransform,
)
from pytorch_ssd.modeling.backbones import backbones
from pytorch_ssd.modeling.box_predictors import box_predictors
from pytorch_ssd.modeling.loss import MultiBoxLoss
from pytorch_ssd.modeling.metrics import MeanAveragePrecision
from pytorch_ssd.modeling.visualize import get_boxes

logger = logging.getLogger(__name__)


class SSD(pl.LightningModule):
    """SSD Detector class."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        learning_rate: float = 1e-3,
        warm_restart_epochs: float = 1 / 3,
        warm_restart_len_mult: int = 2,
        auto_lr_find: bool = False,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        n_classes: Optional[int] = None,
        flip_train: bool = False,
        augment_colors_train: bool = False,
        backbone_name: str = "VGG300",
        use_pretrained_backbone: bool = False,
        predictor_name: str = "SSD",
        image_size: Tuple[int, int] = (300, 300),
        pixel_mean: Tuple[float, ...] = (0.0, 0.0, 0.0),
        pixel_std: Tuple[float, ...] = (1.0, 1.0, 1.0),
        center_variance: float = 0.1,
        size_variance: float = 0.2,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.45,
        max_per_image: int = 100,
        negative_positive_ratio: float = 3,
        calculate_map: bool = True,
        map_iou_threshold: float = 0.5,
        visualize: bool = True,
        **_kwargs,
    ):
        """
        :param dataset_name: used dataset name
        :param data_dir: dataset data directory path
        :param learning_rate: learning rate
        :param lr_reduce_patience: learning rate reduce on plateau patience (epochs)
        :param auto_lr_find: perform auto lr finding
        :param batch_size: mini-batch size for training
        :param num_workers: number of workers for dataloader
        :param pin_memory: pin memory for training
        :param n_classes: number of classes (if 2 then no classification),
            defaults to number of classes in the dataset
        :param flip_train: perform random flipping on train images
        :param augment_colors_train: perform random colors augmentation on train images
        :param backbone_name: used backbone name
        :param use_pretrained_backbone: download pretrained weights for backbone
        :param predictor_name: used predictor name
        :param image_size: image size tuple
        :param pixel_mean: data pixel mean per channel
        :param pixel_std: data pixel std per channel
        :param center_variance: SSD center variance
        :param size_variance: SSD size variance
        :param iou_threshold: IOU threshold for anchors
        :param confidence_threshold: min prediction confidence to use as detection
        :param nms_threshold: non-max suppression IOU threshold
        :param max_per_image: max number of detections per image
        :param negative_positive_ratio: the ratio between the negative examples and
            positive examples for calculating loss
        :param calculate_map: calculate mean average precision during training
        :param map_iou_threshold: mean average precision iou threshold
        :param visualize: perform visualization during training
        """
        super().__init__()
        self.dataset = datasets[dataset_name]
        self.data_dir = data_dir
        if n_classes is None:
            n_classes = len(self.dataset.CLASS_LABELS) + 1
        self.class_labels = (
            self.dataset.CLASS_LABELS if n_classes != 2 else [self.dataset.OBJECT_LABEL]
        )
        backbone = backbones[backbone_name]
        self.backbone = backbone(use_pretrained=use_pretrained_backbone)
        predictor = box_predictors[predictor_name]
        self.predictor = predictor(
            n_classes=n_classes,
            backbone_out_channels=self.backbone.out_channels,
            backbone_boxes_per_loc=self.backbone.boxes_per_loc,
        )
        self.anchors = process_prior(
            image_size=image_size,
            feature_maps=self.backbone.feature_maps,
            min_sizes=self.backbone.min_sizes,
            max_sizes=self.backbone.max_sizes,
            strides=self.backbone.strides,
            aspect_ratios=self.backbone.aspect_ratios,
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
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_per_image = max_per_image
        self.negative_positive_ratio = negative_positive_ratio

        self.lr = learning_rate
        self.auto_lr_find = auto_lr_find
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.n_classes = n_classes
        self.flip_train = flip_train
        self.augment_colors_train = augment_colors_train
        self.calculate_map = calculate_map
        self.map_iou_threshold = map_iou_threshold
        self.visualize = visualize

        self.map = MeanAveragePrecision(iou_threshold=self.map_iou_threshold)

        self.warm_restart_epochs = warm_restart_epochs
        self.warm_restart_len_mult = warm_restart_len_mult

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        """Add SSD args to parent argument parser."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dataset-name",
            type=str,
            default="MNIST",
            help=f"Used dataset name. Available: {list(datasets.keys())}",
        )
        parser.add_argument(
            "--data-dir", type=str, default="data", help="Dataset files directory"
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=1e-3,
            help="Learning rate used for training the model",
        )
        parser.add_argument(
            "--warm-restart-epochs",
            type=float,
            default=1 / 3,
            help="Number of epochs after which a warm restart is performed",
        )
        parser.add_argument(
            "-warm-restart-len-mult",
            type=int,
            default=2,
            help="Coef to multiply warm restart epochs after each restart",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Mini-batch size used for training the model",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="Number of workers used to load the dataset",
        )
        parser.add_argument(
            "--pin-memory",
            default=True,
            action="store_true",
            help="Pin data in memory while training",
        )
        parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
        parser.add_argument(
            "--n-classes",
            type=int,
            default=None,
            help="Number of classes used for training. "
            "If == 2 then only detection without classification",
        )
        parser.add_argument(
            "--backbone-name",
            type=str,
            default="VGG300",
            help=f"Used backbone name. Available: {list(backbones.keys())}",
        )
        parser.add_argument(
            "--use-pretrained-backbone",
            default=False,
            action="store_true",
            help="Start off from pretrained weights from torchvision",
        )
        parser.add_argument(
            "--no-use-pretrained-backbone",
            dest="use_pretrained_backbone",
            action="store_false",
        )
        parser.add_argument(
            "--predictor-name",
            type=str,
            default="SSD",
            help=f"Used box predictor name. Available: {list(box_predictors.keys())}",
        )
        parser.add_argument(
            "--image-size",
            nargs=2,
            type=int,
            default=[300, 300],
            help="Size of the model input image",
        )
        parser.add_argument(
            "--pixel-mean",
            nargs=3,
            type=float,
            default=[0.0, 0.0, 0.0],
            help="Input image pixel means.",
        )
        parser.add_argument(
            "--pixel-std",
            nargs=3,
            type=float,
            default=[1.0, 1.0, 1.0],
            help="Input image pixel stds.",
        )
        parser.add_argument(
            "--center-variance", type=float, default=0.1, help="SSD box center variance"
        )
        parser.add_argument(
            "--size-variance", type=float, default=0.2, help="SSD box size variance"
        )
        parser.add_argument(
            "--iou-threshold", type=float, default=0.5, help="IOU threshold for anchors"
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.8,
            help="Minimum prediction confidence to approve during inference",
        )
        parser.add_argument(
            "--nms-threshold",
            type=float,
            default=0.45,
            help="Non-max suppression IOU threshold",
        )
        parser.add_argument(
            "--max-per-image",
            type=int,
            default=100,
            help="Max number of detections returned per image during inference",
        )
        parser.add_argument(
            "--negative-positive-ratio",
            type=float,
            default=3,
            help="Ratio between negative and positive examples for loss",
        )
        parser.add_argument(
            "--flip-train",
            default=False,
            action="store_true",
            help="Flip train images during training",
        )
        parser.add_argument("--no-flip-train", dest="flip_train", action="store_false")
        parser.add_argument(
            "--augment-colors-train",
            default=False,
            action="store_true",
            help="Perform random colors augmentation during training",
        )
        parser.add_argument(
            "--no-augment-colors-train",
            dest="augment_colors_train",
            action="store_false",
        )
        parser.add_argument(
            "--calculate-map",
            default=True,
            action="store_true",
            help="Calculate Mean Average Precision during training",
        )
        parser.add_argument(
            "--no-calculate-map",
            dest="calculate_map",
            action="store_false",
        )
        parser.add_argument(
            "--map-iou-threshold",
            type=float,
            default=0.5,
            help="Mean Average Precision metric Intersection over Union threshold",
        )
        parser.add_argument(
            "--visualize",
            default=True,
            action="store_true",
            help="Log visualizations of model predictions",
        )
        parser.add_argument("--no-visualize", dest="visualize", action="store_false")
        return parser

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
        for batch_idx in range(batch_size):
            scores = scores_batches[batch_idx]  # (N, num_classes)
            boxes = boxes_batches[batch_idx]  # (N, 4)
            n_boxes, n_classes = scores.shape

            boxes = boxes.view(n_boxes, 1, 4).expand(n_boxes, n_classes, 4)
            labels = torch.arange(n_classes)
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
            priors=self.anchors.to(bbox_pred.device),
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

    def common_run_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_nb: int,
        stage: str,
    ):
        """Common model running step for training and validation."""
        criterion = MultiBoxLoss(self.negative_positive_ratio)

        images, locations, labels = batch

        features = self.backbone(images)
        cls_logits, bbox_pred = self.predictor(features)

        regression_loss, classification_loss = criterion(
            confidence=cls_logits,
            predicted_locations=bbox_pred,
            labels=labels,
            gt_locations=locations,
        )
        loss = regression_loss + classification_loss

        self.log(f"{stage}_loss", loss, prog_bar=False, logger=True)
        self.log(
            f"{stage}_regression_loss", regression_loss, prog_bar=False, logger=True
        )
        self.log(
            f"{stage}_classification_loss",
            classification_loss,
            prog_bar=False,
            logger=True,
        )

        if batch_nb == 0:  # perform at the beginning of each epoch
            if self.calculate_map:
                predictions = self.process_model_prediction(
                    cls_logits.detach(), bbox_pred.detach()
                )
                ground_truth = self.process_model_prediction(
                    labels.detach(), locations.detach()
                )
                self.logger.experiment.log(
                    {f"{stage}_map": self.map(predictions, ground_truth)},
                    step=self.global_step,
                )

            if self.visualize:
                image = images[0].detach().permute(1, 2, 0)
                denominator = torch.reciprocal(
                    torch.tensor(self.pixel_std, device=self.device)
                )
                image = image / denominator + torch.tensor(
                    self.pixel_mean, device=self.device
                )
                image.clamp_(min=0, max=1)

                ((gt_boxes, gt_scores, gt_labels),) = self.process_model_prediction(
                    labels[0].detach().unsqueeze(0),
                    locations[0].detach().unsqueeze(0),
                    confidence_threshold=0.0,
                )
                ((boxes, scores, labels),) = self.process_model_prediction(
                    cls_logits[0].detach().unsqueeze(0),
                    bbox_pred[0].detach().unsqueeze(0),
                    confidence_threshold=0.0,
                )
                log_boxes = get_boxes(
                    gt_boxes=gt_boxes.cpu(),
                    gt_scores=gt_scores.cpu(),
                    gt_labels=gt_labels.cpu(),
                    boxes=boxes.cpu(),
                    scores=scores.cpu(),
                    labels=labels.cpu(),
                    class_labels=self.class_labels,
                )
                log_image = wandb.Image(
                    image.cpu().numpy(),
                    boxes=log_boxes,
                    caption="object detection predictions",
                )
                self.logger.experiment.log(
                    {f"{stage}_predictions": log_image}, step=self.global_step
                )

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_nb: int
    ):
        """Step for training."""
        return self.common_run_step(batch, batch_nb, stage="train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_nb: int
    ):
        """Step for validation."""
        return self.common_run_step(batch, batch_nb, stage="val")

    def configure_optimizers(self):
        """Configure training optimizer."""
        warm_restart_steps = int(
            len(self.train_dataloader()) * self.warm_restart_epochs
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=warm_restart_steps,
            T_mult=self.warm_restart_len_mult,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def train_dataloader(self) -> DataLoader:
        """Prepare train dataloader."""
        data_transform = TrainDataTransform(
            image_size=self.image_size,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
            flip=self.flip_train,
            augment_colors=self.augment_colors_train,
        )
        dataset = self.dataset(
            self.data_dir,
            data_transform=data_transform,
            target_transform=self.target_transform,
            subset="train",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Prepare validation dataloader."""
        data_transform = DataTransform(
            image_size=self.image_size,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
        )
        dataset = self.dataset(
            self.data_dir,
            data_transform=data_transform,
            target_transform=self.target_transform,
            subset="test",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

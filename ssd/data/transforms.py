"""Data transforms."""
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from albumentations import (
    BasicTransform,
    BboxParams,
    Compose,
    HorizontalFlip,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
    RandomSizedBBoxSafeCrop,
    Resize,
    normalize_bboxes,
)
from albumentations.pytorch import ToTensorV2 as ToTensor
from yacs.config import CfgNode

from ssd.data.bboxes import (
    assign_priors,
    center_bbox_to_corner_bbox,
    convert_boxes_to_locations,
    corner_bbox_to_center_bbox,
)
from ssd.data.priors import process_prior


class SSDTargetTransform:
    """Transforms for SSD target."""

    def __init__(self, config: CfgNode):
        self.center_form_priors = process_prior(
            image_size=config.DATA.SHAPE,
            feature_maps=config.DATA.PRIOR.FEATURE_MAPS,
            min_sizes=config.DATA.PRIOR.MIN_SIZES,
            max_sizes=config.DATA.PRIOR.MAX_SIZES,
            strides=config.DATA.PRIOR.STRIDES,
            aspect_ratios=config.DATA.PRIOR.ASPECT_RATIOS,
            clip=config.DATA.PRIOR.CLIP,
        )
        self.corner_form_priors = center_bbox_to_corner_bbox(self.center_form_priors)
        self.center_variance = config.MODEL.CENTER_VARIANCE
        self.size_variance = config.MODEL.SIZE_VARIANCE
        self.iou_threshold = config.MODEL.IOU_THRESHOLD
        self.image_shape = config.DATA.SHAPE
        self.single_class = config.DATA.N_CLASSES == 2

    def __call__(
        self,
        gt_boxes: Union[np.ndarray, torch.Tensor],
        gt_labels: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = assign_priors(
            gt_boxes,
            gt_labels,
            corner_form_priors=self.corner_form_priors,
            iou_threshold=self.iou_threshold,
        )
        boxes = corner_bbox_to_center_bbox(boxes)
        locations = convert_boxes_to_locations(
            boxes,
            self.center_form_priors,
            center_variance=self.center_variance,
            size_variance=self.size_variance,
        )
        if self.single_class:
            labels[labels > 0] = 1
        return locations, labels


class DataTransform:
    """Base class for image transforms using albumentations."""

    def __init__(
        self, config: CfgNode, transforms: Optional[List[BasicTransform]] = None
    ):
        if transforms is None:
            transforms = []
        self.transforms = transforms
        default_transforms = [
            Resize(*config.DATA.SHAPE),
            Normalize(mean=config.DATA.PIXEL_MEAN, std=config.DATA.PIXEL_STD,),
            ToTensor(),
        ]
        self.transforms.extend(default_transforms)
        self.normalize_bboxes = partial(
            normalize_bboxes, rows=config.DATA.SHAPE[0], cols=config.DATA.SHAPE[1]
        )

    def __call__(
        self,
        image: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        image = image.permute(1, 2, 0)
        if bboxes is not None and labels is not None:
            augment = Compose(
                self.transforms,
                bbox_params=BboxParams(
                    format="pascal_voc", label_fields=["labels"], min_visibility=0.2
                ),
            )
            augmented = augment(image=image.numpy(), bboxes=bboxes, labels=labels)
            image = augmented["image"]
            bboxes = torch.tensor(
                self.normalize_bboxes(augmented["bboxes"]), dtype=torch.float32
            )
            labels = torch.tensor(augmented["labels"])
        else:
            augment = Compose(self.transforms)
            image = augment(image=image.numpy())["image"]
        return image, bboxes, labels


class TrainDataTransform(DataTransform):
    """Transforms images and labels for training SSD."""

    def __init__(self, config: CfgNode, flip: bool = False):
        transforms = []
        if config.DATA.CHANNELS == 3:
            # noinspection PyTypeChecker
            color_transforms = [
                HueSaturationValue(
                    hue_shift_limit=0.05, sat_shift_limit=0.5, val_shift_limit=0.2
                ),
                RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.5),
            ]
            transforms.extend(color_transforms)
        shape_transforms = [
            HorizontalFlip(p=flip * 0.5),
            RandomSizedBBoxSafeCrop(512, 512),
        ]
        transforms.extend(shape_transforms)
        super().__init__(config, transforms)

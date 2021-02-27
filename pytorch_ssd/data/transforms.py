"""Data transforms."""
from typing import List, Optional, Tuple, Union

import cv2
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
    RandomCrop,
    RandomSizedBBoxSafeCrop,
    Resize,
    get_random_crop_coords,
    normalize_bboxes,
)
from albumentations.pytorch import ToTensorV2 as ToTensor

from pytorch_ssd.data.bboxes import (
    assign_priors,
    center_bbox_to_corner_bbox,
    convert_boxes_to_locations,
    corner_bbox_to_center_bbox,
)


class SSDTargetTransform:
    """Transforms for SSD target."""

    def __init__(
        self,
        anchors: torch.Tensor,
        image_size: Tuple[int, int],
        n_classes: int,
        center_variance: float,
        size_variance: float,
        iou_threshold: float,
        drop_small_boxes: bool = False,
    ):
        self.center_form_priors = anchors
        self.corner_form_priors = center_bbox_to_corner_bbox(self.center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.image_shape = image_size
        self.single_class = n_classes == 2
        self.drop = drop_small_boxes

    def __call__(
        self,
        gt_boxes: Union[np.ndarray, torch.Tensor],
        gt_labels: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        if gt_labels.numel() == 0:
            return torch.tensor([]), torch.tensor([])
        if self.drop:
            boxes_mask_w = gt_boxes[:, 2] - gt_boxes[:, 0] > 0.04
            boxes_mask_h = gt_boxes[:, 3] - gt_boxes[:, 1] > 0.04
            boxes_mask = torch.logical_and(boxes_mask_w, boxes_mask_h)
            gt_boxes = gt_boxes[boxes_mask.unsqueeze(-1).expand_as(gt_boxes)].view(
                -1, 4
            )
            gt_labels = gt_labels[boxes_mask]
        if gt_labels.numel() == 0:
            return torch.tensor([]), torch.tensor([])
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
        self,
        image_size: Tuple[int, int],
        pixel_mean: List[float],
        pixel_std: List[float],
        transforms: Optional[List[BasicTransform]] = None,
    ):
        """
        :param image_size: model input data shape (eg. (300, 300))
        :param pixel_mean: data pixel mean per channel
        :param pixel_std: data pixel std per channel
        :param transforms: initial transforms
        """
        if transforms is None:
            transforms = []
        self.transforms = transforms
        default_transforms = [
            Resize(*image_size),
            Normalize(
                mean=pixel_mean,
                std=pixel_std,
                max_pixel_value=1.0,
            ),
            ToTensor(),
        ]
        self.transforms.extend(default_transforms)

    def __call__(
        self,
        image: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if bboxes is not None and labels is not None:
            augment = Compose(
                self.transforms,
                bbox_params=BboxParams(
                    format="pascal_voc", label_fields=["labels"], min_visibility=0.2
                ),
            )
            augmented = augment(image=image.numpy(), bboxes=bboxes, labels=labels)
            image = augmented["image"]
            _, height, width = image.shape
            bboxes = torch.tensor(
                normalize_bboxes(augmented["bboxes"], rows=height, cols=width),
                dtype=torch.float32,
            )
            labels = torch.tensor(augmented["labels"])
        else:
            augment = Compose(self.transforms)
            image = augment(image=image.numpy())["image"]
        return image, bboxes, labels


class TrainDataTransform(DataTransform):
    """Transforms images and labels for training SSD."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        pixel_mean: List[float],
        pixel_std: List[float],
        flip: bool = False,
        augment_colors: bool = False,
        strong_crop: bool = False,
    ):
        """
        :param image_size: model input data shape (eg. (300, 300))
        :param pixel_mean: data pixel mean per channel
        :param pixel_std: data pixel std per channel
        :param flip: randomly flip image L/R
        :param augment_colors: augment image colors
        :param strong_crop: crop input image to smaller size regardless of boxes
        """
        transforms = [
            HorizontalFlip(p=flip * 0.5),
            RandomSizedBBoxSafeCrop(
                int(2 * image_size[1]), int(2 * image_size[0]), erosion_rate=0.2
            ),
        ]
        if strong_crop:
            transforms.append(RandomCrop(image_size[1], image_size[0]))
        if augment_colors:
            # noinspection PyTypeChecker
            color_transforms = [
                HueSaturationValue(
                    hue_shift_limit=0.05, sat_shift_limit=0.5, val_shift_limit=0.2
                ),
                RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.5),
            ]
            transforms.extend(color_transforms)
        super().__init__(
            image_size=image_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            transforms=transforms,
        )

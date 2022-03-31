"""SSD box predictors."""
from typing import List, Tuple

import torch
import torch.nn as nn


class BaseBoxPredictor(nn.Module):
    """Base class for box predictor."""

    def __init__(
        self,
        n_classes: int,
        backbone_out_channels: List[int],
        backbone_boxes_per_loc: List[int],
    ):
        super().__init__()
        self.n_classes = n_classes
        self.backbone_out_channels = backbone_out_channels
        self.backbone_boxes_per_loc = backbone_boxes_per_loc
        self.cls_headers = self._build_cls_headers()
        self.reg_headers = self._build_reg_headers()
        self.reset_params()

    def _build_cls_headers(self) -> nn.ModuleList:
        """Build class logits headers module."""
        raise NotImplementedError()

    def _build_reg_headers(self) -> nn.ModuleList:
        """Build bbox regression headers module."""
        raise NotImplementedError()

    def reset_params(self):
        """Initialize model params."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_logits = []
        bbox_pred = []
        batch_size = features[0].shape[0]
        for feature, cls_header, reg_header in zip(
            features, self.cls_headers, self.reg_headers
        ):
            cls_logits.append(
                cls_header(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, self.n_classes)
            )
            bbox_pred.append(
                reg_header(feature)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, -1, 4)
            )

        cls_logits = torch.cat(cls_logits, dim=1)
        bbox_pred = torch.cat(bbox_pred, dim=1)

        return cls_logits, bbox_pred


class SSDBoxPredictor(BaseBoxPredictor):
    """SSD Box Predictor."""

    def _build_cls_headers(self) -> nn.ModuleList:
        """Build SSD cls headers."""
        layers = [
            nn.Conv2d(
                in_channels=channels,
                out_channels=boxes * self.n_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for boxes, channels in zip(
                self.backbone_boxes_per_loc, self.backbone_out_channels
            )
        ]
        return nn.ModuleList(layers)

    def _build_reg_headers(self) -> nn.ModuleList:
        """Build SSD bbox pred headers."""
        layers = [
            nn.Conv2d(
                in_channels=channels,
                out_channels=boxes * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for boxes, channels in zip(
                self.backbone_boxes_per_loc, self.backbone_out_channels
            )
        ]
        return nn.ModuleList(layers)

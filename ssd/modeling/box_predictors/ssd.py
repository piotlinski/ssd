"""SSD box predictors."""
from typing import List, Tuple

import torch
import torch.nn as nn
from yacs.config import CfgNode


class BoxPredictor(nn.Module):
    """Base class for box predictor."""

    def __init__(self, config: CfgNode, backbone_out_channels: List[int]):
        super().__init__()
        self.backbone_out_channels = backbone_out_channels
        self.config = config
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
        for feature, cls_header, reg_header in zip(
            features, self.cls_headers, self.reg_headers
        ):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat(
            [logit.view(logit.shape[0], -1) for logit in cls_logits], dim=1
        ).view(batch_size, -1, self.config.DATA.N_CLASSES)
        bbox_pred = torch.cat(
            [reg.view(reg.shape[0], -1) for reg in bbox_pred], dim=1
        ).view(batch_size, -1, 4)

        return cls_logits, bbox_pred


class SSDBoxPredictor(BoxPredictor):
    """SSD Box Predictor."""

    def _build_cls_headers(self) -> nn.ModuleList:
        """Build SSD cls headers."""
        layers = [
            nn.Conv2d(
                in_channels=channels,
                out_channels=boxes * self.config.DATA.N_CLASSES,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for boxes, channels in zip(
                self.config.DATA.PRIOR.BOXES_PER_LOC, self.backbone_out_channels
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
                self.config.DATA.PRIOR.BOXES_PER_LOC, self.backbone_out_channels
            )
        ]
        return nn.ModuleList(layers)

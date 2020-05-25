"""SSD box predictors."""
from typing import Callable, Iterable, List, Tuple

import torch
import torch.nn as nn
from yacs.config import CfgNode


class BoxPredictor(nn.Module):
    """Base class for box predictor."""

    def __init__(
        self,
        config: CfgNode,
        backbone_out_channels: List[int],
        boxes_per_location: List[int],
    ):
        super().__init__()
        self.backbone_out_channels = backbone_out_channels
        self.config = config
        self.cls_headers = nn.ModuleList(
            list(
                self._headers(
                    block=self.cls_block,
                    boxes_per_location_list=boxes_per_location,
                    out_channels_list=backbone_out_channels,
                )
            )
        )
        self.reg_headers = nn.ModuleList(
            list(
                self._headers(
                    block=self.reg_block,
                    boxes_per_location_list=boxes_per_location,
                    out_channels_list=backbone_out_channels,
                )
            )
        )
        self.reset_params()

    @staticmethod
    def _headers(
        block: Callable[[int, int, int], nn.Module],
        boxes_per_location_list: List[int],
        out_channels_list: List[int],
    ) -> Iterable[nn.Module]:
        """Prepare single header"""
        for level, (out_channels, boxes_per_location) in enumerate(
            zip(out_channels_list, boxes_per_location_list)
        ):
            yield block(level, out_channels, boxes_per_location)

    def cls_block(
        self, level: int, out_channels: int, boxes_per_location: int
    ) -> nn.Module:
        """Single class prediction block."""
        raise NotImplementedError

    def reg_block(
        self, level: int, out_channels: int, boxes_per_location: int
    ) -> nn.Module:
        """Single bbox regression block."""
        raise NotImplementedError

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

    def cls_block(
        self, level: int, out_channels: int, boxes_per_location: int
    ) -> nn.Module:
        """SSD Box Predictor class block."""
        return nn.Conv2d(
            in_channels=out_channels,
            out_channels=boxes_per_location * self.config.DATA.N_CLASSES,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def reg_block(
        self, level: int, out_channels: int, boxes_per_location: int
    ) -> nn.Module:
        """SSD Box Predictor bbox block."""
        return nn.Conv2d(
            in_channels=out_channels,
            out_channels=boxes_per_location * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )

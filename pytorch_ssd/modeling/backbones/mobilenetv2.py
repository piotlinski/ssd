"""MobileNetV2 backbone for SSD."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models.mobilenet import mobilenet_v2

from pytorch_ssd.modeling.backbones.base import BaseBackbone


class MobileNetV2(BaseBackbone):
    """MobileNetV2 backbone for SSD."""

    def __init__(self, use_pretrained: bool, **kwargs):
        self.out_channels = kwargs.get("out_channels") or [1280, 640, 640, 320, 320]
        self.feature_maps = kwargs.get("feature_maps") or [1, 3, 5, 11, 23]
        self.min_sizes = kwargs.get("min_sizes") or [251, 187, 123, 59, 21]
        self.max_sizes = kwargs.get("max_sizes") or [315, 251, 187, 123, 59]
        self.strides = kwargs.get("strides") or [300, 100, 64, 32, 16]
        self.aspect_ratios = kwargs.get("aspect_ratios") or [(), (), (), (), ()]
        super().__init__(
            use_pretrained=use_pretrained,
        )

    def _build_backbone(self) -> nn.Module:
        """Build MobileNetV2 backbone."""
        backbone = mobilenet_v2(pretrained=self.use_pretrained).features
        backbone.add_module("avgpool", nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        return backbone

    def _build_extras(self) -> nn.Module:
        """Build MobileNetV2 extras."""
        layers = [
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
            )
            for in_channels, out_channels in zip(
                self.out_channels, self.out_channels[1:]
            )
        ]
        extras = nn.ModuleList(layers)
        return extras

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run data through MobileNetV2 backbone."""
        features = []

        x = self.backbone(x)
        features.append(x)

        for layer in self.extras:
            x = functional.relu(layer(x), inplace=True)
            features.append(x)

        return tuple(features)

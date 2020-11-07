"""MobileNetV2 backbone for SSD."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models.mobilenet import mobilenet_v2

from pyssd.modeling.backbones.base import BaseBackbone


class MobileNetV2(BaseBackbone):
    """MobileNetV2 backbone for SSD."""

    def __init__(self, use_pretrained: bool):
        super().__init__(
            out_channels=[1280, 640, 640, 320, 320],
            feature_maps=[1, 3, 5, 11, 23],
            min_sizes=[251, 187, 123, 59, 21],
            max_sizes=[315, 251, 187, 123, 59],
            strides=[300, 100, 64, 32, 16],
            aspect_ratios=[(), (), (), (), ()],
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
                in_channels=1280, out_channels=640, kernel_size=3, stride=2
            ),
            nn.ConvTranspose2d(
                in_channels=640, out_channels=640, kernel_size=3, stride=1
            ),
            nn.ConvTranspose2d(
                in_channels=640, out_channels=320, kernel_size=3, stride=2
            ),
            nn.ConvTranspose2d(
                in_channels=320, out_channels=320, kernel_size=3, stride=2
            ),
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

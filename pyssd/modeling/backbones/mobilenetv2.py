"""MobileNetV2 backbone for SSD."""
from typing import Tuple

from torchvision.models.mobilenet import mobilenet_v2

import torch
import torch.nn as nn
from pyssd.modeling.backbones.base import BaseBackbone


class MobileNetV2(BaseBackbone):
    """MobileNetV2 backbone for SSD."""

    def _build_backbone(self) -> nn.Module:
        """Build MobileNetV2 backbone."""

    def _build_extras(self) -> nn.Module:
        """Build MobileNetV2 300x300 extras."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run data through VGG11 backbone."""

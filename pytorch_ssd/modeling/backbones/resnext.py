"""ResNeXt backbone for SSD."""
from abc import ABC
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models import resnext50_32x4d, resnext101_32x8d

from pytorch_ssd.modeling.backbones import BaseBackbone

resnext_backbones: Dict[int, Callable[..., nn.Module]] = {
    50: resnext50_32x4d,
    101: resnext101_32x8d,
}


class ResNeXt(BaseBackbone, ABC):
    """ResNeXt SSD meta backbone."""

    def __init__(
        self,
        out_channels: List[int],
        feature_maps: List[int],
        min_sizes: List[float],
        max_sizes: List[float],
        strides: List[int],
        aspect_ratios: List[Tuple[int, ...]],
        use_pretrained: bool,
        resnext_size: int,
    ):
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.strides = strides
        self.aspect_ratios = aspect_ratios
        self._resnext = resnext_backbones[resnext_size]
        super().__init__(use_pretrained)

    def _build_backbone(self) -> nn.Module:
        """Build ResNeXt backbone."""
        backbone = self._resnext(pretrained=self.use_pretrained)
        return nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run data through ResNeXt backbone."""
        features = []
        out_idx = {6, 7}

        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in out_idx:
                features.append(x)

        for idx, layer in enumerate(self.extras):
            x = functional.relu(layer(x), inplace=True)
            if idx % 2 == 1:
                features.append(x)

        return tuple(features)


class ResNeXt300(ResNeXt, ABC):
    """ResNeXt backbone for 300x300 input."""

    def __init__(self, use_pretrained: bool, resnext_size: int, **kwargs):
        super().__init__(
            out_channels=kwargs.get("out_channels") or [1024, 2048, 2048, 2048, 2048],
            feature_maps=kwargs.get("feature_maps") or [19, 10, 5, 3, 1],
            min_sizes=kwargs.get("min_sizes") or [32, 80, 153, 207, 261],
            max_sizes=kwargs.get("max_sizes") or [80, 153, 207, 261, 315],
            strides=kwargs.get("strides") or [16, 32, 64, 100, 300],
            aspect_ratios=kwargs.get("aspect_ratios") or [(), (), (), (), ()],
            use_pretrained=use_pretrained,
            resnext_size=resnext_size,
        )

    def _build_extras(self) -> nn.Module:
        """Build extras for 300x300 input."""
        extras = [
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.Conv2d(
                in_channels=1024, out_channels=2048, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.Conv2d(
                in_channels=1024, out_channels=2048, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3),
        ]
        return nn.ModuleList(extras)


class ResNeXt50_300(ResNeXt300):
    """ResNeXt50 backbone for 300x300 input."""

    def __init__(self, use_pretrained: bool, **kwargs):
        super().__init__(use_pretrained=use_pretrained, resnext_size=50, **kwargs)


class ResNeXt101_300(ResNeXt300):
    """ResNeXt101 backbone for 300x300 input."""

    def __init__(self, use_pretrained: bool, **kwargs):
        super().__init__(use_pretrained=use_pretrained, resnext_size=101, **kwargs)

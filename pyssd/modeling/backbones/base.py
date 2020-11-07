"""Base class for SSD backbone."""
from typing import List, Tuple

import torch
import torch.nn as nn


class BaseBackbone(nn.Module):
    def __init__(
        self,
        out_channels: List[int],
        feature_maps: List[int],
        min_sizes: List[float],
        max_sizes: List[float],
        strides: List[int],
        aspect_ratios: List[Tuple[int, ...]],
        use_pretrained: bool,
    ):
        """
        :param out_channels: backbone output channels
        :param feature_maps: number of features in each output map
        :param min_sizes: minimum object size in each feature map
        :param max_sizes: maximum object size in each feature map
        :param strides: stride in each feature map
        :param aspect_ratios: additional rectangular boxes in each feature map
        :param use_pretrained: use pretrained backbone
        """
        super().__init__()

        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.strides = strides
        self.aspect_ratios = aspect_ratios
        self.boxes_per_loc = [
            2 + 2 * len(aspect_ratio_tuple) for aspect_ratio_tuple in self.aspect_ratios
        ]
        self.use_pretrained = use_pretrained

        self.backbone = self._build_backbone()
        self.extras = self._build_extras()
        self.init_extras()

    def init_extras(self):
        """Initialize model params."""
        for module in self.extras.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_backbone(self) -> nn.Module:
        """Build backbone (that may be pretrained)."""
        raise NotImplementedError()

    def _build_extras(self) -> nn.Module:
        """Build backbone extras for creating features."""
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run input through backbone to get features."""
        raise NotImplementedError()

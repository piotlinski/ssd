"""Base class for SSD backbone."""
from abc import ABC
from typing import List, Tuple

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    def __init__(self, use_pretrained: bool):
        """
        :param use_pretrained: use pretrained backbone
        """
        super().__init__()

        self.out_channels: List[int]
        self.feature_maps: List[int]
        self.min_sizes: List[float]
        self.max_sizes: List[float]
        self.strides: List[int]
        self.aspect_ratios: List[Tuple[int, ...]]
        self.use_pretrained = use_pretrained

        self.backbone = self._build_backbone()
        self.extras = self._build_extras()
        self.init_extras()

    @property
    def boxes_per_loc(self) -> List[int]:
        return [
            2 + 2 * len(aspect_ratio_tuple) for aspect_ratio_tuple in self.aspect_ratios
        ]

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

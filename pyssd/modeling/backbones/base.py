"""Base class for SSD backbone."""
from typing import List, Tuple

import torch
import torch.nn as nn
from yacs.config import CfgNode


class BaseBackbone(nn.Module):
    def __init__(self, config: CfgNode, out_channels: List[int]):
        """
        :param config: SSD config
        :param out_channels: output channels of the backbone
        """
        super().__init__()
        self.config = config
        self.backbone = self._build_backbone()
        self.extras = self._build_extras()
        self.out_channels = out_channels
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

"""Base class for SSD backbone."""
from typing import List, Tuple

import torch
import torch.nn as nn
from yacs.config import CfgNode

from ssd.modeling.checkpoint import cache_url


class BaseBackbone(nn.Module):
    def __init__(self, config: CfgNode, out_channels: List[int]):
        """
        :param config: SSD config
        :param out_channels: output channels of the backbone
        """
        super().__init__()
        self.backbone = self._build_backbone()
        self.extras = self._build_extras()
        self.out_channels = out_channels
        self.reset_params()
        if config.MODEL.PRETRAINED_URL:
            self.init_pretrain(
                url=config.MODEL.PRETRAINED_URL,
                pretrained_directory=(
                    f"{config.ASSETS_DIR}/{config.MODEL.PRETRAINED_DIR}"
                ),
            )
        else:
            self.init_xavier()

    def reset_params(self):
        """Initialize model params."""
        for module in self.extras.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def init_pretrain(self, url: str, pretrained_directory: str):
        """Initialize from a downloaded pretrained model."""
        cached_file = cache_url(url, pretrained_directory)
        state_dict = torch.load(cached_file, map_location="cpu")
        self.backbone.load_state_dict(state_dict)

    def init_xavier(self):
        """Initialize backbone using xavier initializer."""
        for module in self.backbone.modules():
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

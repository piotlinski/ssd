"""VGG backbone for SSD."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import init
from torchvision.models.vgg import vgg11, vgg11_bn, vgg16, vgg16_bn
from yacs.config import CfgNode

from pyssd.modeling.backbones.base import BaseBackbone
from pyssd.modeling.checkpoint import cache_url


class L2Norm(nn.Module):
    """L2 Norm layer."""

    def __init__(self, n_channels: int, scale: Optional[float]):
        super().__init__()
        self.n_channels: int = n_channels
        self.gamma: Optional[float] = scale
        self.eps: float = 1e-10
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(self.n_channels), requires_grad=True
        )
        self.reset_params()

    def reset_params(self):
        """Set weights according to scale."""
        init.constant_(self.weight, self.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x


class VGG11(BaseBackbone):
    """VGG11 light backbone. Allows 300x300 input."""

    def __init__(self, config: CfgNode):
        super().__init__(config=config, out_channels=[512, 512, 512, 256, 256])

    def _build_backbone(self) -> nn.Module:
        """Build VGG11 backbone."""
        torchvision_pretrained = (
            self.config.MODEL.USE_PRETRAINED and not self.config.MODEL.PRETRAINED_URL
        )
        if self.config.MODEL.BATCH_NORM:
            backbone = vgg11_bn(pretrained=torchvision_pretrained).features
        else:
            backbone = vgg11(pretrained=torchvision_pretrained).features
        return backbone

    def _build_extras(self) -> nn.Module:
        """Build VGG11 300x300 extras."""
        layers = [
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
        ]
        extras = nn.ModuleList(layers)
        return extras

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run data through VGG11 backbone."""
        features = []
        maxpools = 0
        mid_done = False

        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                maxpools += 1
            if maxpools == 4 and not mid_done:
                features.append(x)
                mid_done = True

        features.append(x)  # vgg output

        for idx, layer in enumerate(self.extras):
            x = nn.functional.relu(layer(x), inplace=True)
            if idx % 2 == 1:
                features.append(x)  # each SSD feature

        return tuple(features)


class VGG16(BaseBackbone):
    """VGG16 backbone. Allows 300x300 or 512x512 input."""

    def __init__(self, config: CfgNode):
        if config.DATA.SHAPE == (300, 300):
            out_channels = [512, 1024, 512, 256, 256, 256]
        elif config.DATA.SHAPE == (512, 512):
            out_channels = [512, 1024, 512, 256, 256, 256, 256]
        else:
            raise ValueError("Data shape must be either 300x300 or 512x512.")
        super().__init__(config=config, out_channels=out_channels)
        self.l2_norm = L2Norm(n_channels=512, scale=20)

    def _build_backbone(self) -> nn.Module:
        """Build VGG16 backbone."""
        torchvision_pretrained = (
            self.config.MODEL.USE_PRETRAINED and not self.config.MODEL.PRETRAINED_URL
        )
        if self.config.MODEL.BATCH_NORM:
            backbone = vgg16_bn(pretrained=torchvision_pretrained).features[:-1]
            backbone[23].ceil_mode = True
        else:
            backbone = vgg16(pretrained=torchvision_pretrained).features[:-1]
            backbone[16].ceil_mode = True
        start_id = len(backbone)
        backbone.add_module(
            f"{start_id}", nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        backbone.add_module(
            f"{start_id + 1}",
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6
            ),
        )
        backbone.add_module(f"{start_id +2}", nn.ReLU(inplace=True))
        backbone.add_module(
            f"{start_id + 3}",
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
        )
        backbone.add_module(f"{start_id + 4}", nn.ReLU(inplace=True))
        if self.config.MODEL.USE_PRETRAINED and self.config.MODEL.PRETRAINED_URL:
            cached_file = cache_url(
                self.config.MODEL.PRETRAINED_URL,
                f"{self.config.ASSETS_DIR}/{self.config.MODEL.PRETRAINED_DIR}",
            )
            state_dict = torch.load(cached_file, map_location="cpu")
            backbone.load_state_dict(state_dict)
        else:
            for module in backbone[start_id:]:
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        return backbone

    def _build_extras(self) -> nn.Module:
        """Build extras for 300x300 input."""
        layers = [
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
        ]
        if self.config.DATA.SHAPE == (300, 300):
            layers.extend(
                [
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                ]
            )
        elif self.config.DATA.SHAPE == (512, 512):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.Conv2d(
                        in_channels=256, out_channels=128, kernel_size=1, stride=1
                    ),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=4,
                        stride=1,
                        padding=1,
                    ),
                ]
            )
        extras = nn.ModuleList(layers)
        return extras

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Run data through VGG16 backbone."""
        features = []
        relus = 0
        l2norm_done = False

        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                relus += 1
            if relus == 10 and not l2norm_done:
                features.append(self.l2_norm(x))  # conv4_3 L2 norm
                l2norm_done = True

        features.append(x)  # vgg output

        for idx, layer in enumerate(self.extras):
            x = nn.functional.relu(layer(x), inplace=True)
            if idx % 2 == 1:
                features.append(x)  # each SSD feature

        return tuple(features)

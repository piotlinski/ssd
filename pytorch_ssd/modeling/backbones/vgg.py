"""VGG backbone for SSD."""
from abc import ABC
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import init
from torchvision.models.vgg import vgg11, vgg11_bn, vgg16, vgg16_bn

from pytorch_ssd.modeling.backbones.base import BaseBackbone


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


class VGGLite(BaseBackbone):
    """VGG11 light backbone."""

    def __init__(self, use_pretrained: bool):
        self.out_channels = [512, 512, 512, 256, 256]
        self.feature_maps = [18, 9, 5, 3, 1]
        self.min_sizes = [32, 80, 153, 207, 261]
        self.max_sizes = [80, 153, 207, 261, 315]
        self.strides = [16, 32, 64, 100, 300]
        self.aspect_ratios = [(), (), (), (), ()]
        super().__init__(
            use_pretrained=use_pretrained,
        )

    def _build_backbone(self) -> nn.Module:
        """Build VGG11 backbone."""
        return vgg11(pretrained=self.use_pretrained).features

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
            x = functional.relu(layer(x), inplace=True)
            if idx % 2 == 1:
                features.append(x)  # each SSD feature

        return tuple(features)


class VGGLiteBN(VGGLite):
    """Batch Norm version of VGGLite."""

    def _build_backbone(self) -> nn.Module:
        """Build VGG11_BN backbone."""
        return vgg11_bn(pretrained=self.use_pretrained).features


class VGG16(BaseBackbone, ABC):
    """VGG16 backbone."""

    def __init__(
        self,
        out_channels: List[int],
        feature_maps: List[int],
        min_sizes: List[float],
        max_sizes: List[float],
        strides: List[int],
        aspect_ratios: List[Tuple[int, ...]],
        use_pretrained: bool,
        batch_norm: bool,
    ):
        self.batch_norm = batch_norm
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.strides = strides
        self.aspect_ratios = aspect_ratios
        super().__init__(
            use_pretrained=use_pretrained,
        )
        self.l2_norm = L2Norm(n_channels=512, scale=20)

    def _build_backbone(self) -> nn.Module:
        """Build VGG16 backbone."""
        if self.batch_norm:
            backbone = vgg16_bn(pretrained=self.use_pretrained).features[:-1]
            backbone[23].ceil_mode = True
        else:
            backbone = vgg16(pretrained=self.use_pretrained).features[:-1]
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
        for module in backbone[start_id:]:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        return backbone

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


class VGG300(VGG16):
    """VGG16 backbone for 300x300 input."""

    def __init__(self, use_pretrained: bool, batch_norm: bool = False):
        super().__init__(
            batch_norm=batch_norm,
            out_channels=[512, 1024, 512, 256, 256, 256],
            feature_maps=[38, 19, 10, 5, 3, 1],
            min_sizes=[21, 45, 99, 153, 207, 261],
            max_sizes=[45, 99, 153, 207, 261, 315],
            strides=[8, 16, 32, 64, 100, 300],
            aspect_ratios=[(2,), (2, 3), (2, 3), (2, 3), (2,), (2,)],
            use_pretrained=use_pretrained,
        )

    def _build_extras(self) -> nn.Module:
        """Build extras for 300x300 input."""
        extras = [
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
        ]
        return nn.ModuleList(extras)


class VGG300BN(VGG300):
    """Batch Norm version of VGG300."""

    def __init__(self, use_pretrained: bool):
        super().__init__(use_pretrained=use_pretrained, batch_norm=True)


class VGG512(VGG16):
    """VGG16 backbone for 512x512 input."""

    def __init__(self, use_pretrained: bool, batch_norm: bool = False):
        super().__init__(
            batch_norm=batch_norm,
            out_channels=[512, 1024, 512, 256, 256, 256, 256],
            feature_maps=[64, 32, 16, 8, 4, 2, 1],
            min_sizes=[20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
            max_sizes=[51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
            strides=[8, 16, 32, 64, 128, 256, 512],
            aspect_ratios=[(2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,)],
            use_pretrained=use_pretrained,
        )

    def _build_extras(self) -> nn.Module:
        """Build extras for 300x300 input."""
        extras = [
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1
            ),
        ]
        return nn.ModuleList(extras)


class VGG512BN(VGG512):
    """Batch Norm version of VGG512."""

    def __init__(self, use_pretrained: bool):
        super().__init__(use_pretrained=use_pretrained, batch_norm=True)

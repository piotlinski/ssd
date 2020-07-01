"""VGG backbone for SSD."""
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import init
from yacs.config import CfgNode

from ssd.modeling.checkpoint import cache_url


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


class VGG(nn.Module):
    """Base class for VGG backbone."""

    def __init__(
        self,
        config: CfgNode,
        vgg_config: List[Union[str, int]],
        vgg_extras_config: List[Union[str, int]],
        out_channels: List[int],
    ):
        super().__init__()
        self.backbone = nn.ModuleList(
            list(
                self._vgg(
                    vgg_config=vgg_config,
                    in_channels=config.DATA.CHANNELS,
                    batch_norm=config.MODEL.BATCH_NORM,
                )
            )
        )
        self.extras = nn.ModuleList(
            list(
                self._vgg_extras(vgg_extras_config=vgg_extras_config, in_channels=1024,)
            )
        )
        self.out_channels = out_channels
        self.l2_norm = L2Norm(512, scale=20)
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
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def init_pretrain(self, url: str, pretrained_directory: str):
        """Initialize from a downloaded pretrained model."""
        cached_file = cache_url(url, pretrained_directory)
        state_dict = torch.load(cached_file, map_location="cpu")
        self.backbone.load_state_dict(state_dict)

    def init_xavier(self):
        """Initialize backbone using xavier initializer."""
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        l2_done = False
        features = []
        convs = 0
        for layer in self.backbone:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                convs += 1
            if convs == 10 and not l2_done:
                features.append(self.l2_norm(x))  # conv4_3 L2 norm
                l2_done = True

        features.append(x)  # vgg output

        for idx, layer in enumerate(self.extras):
            x = nn.ReLU(inplace=True)(layer(x))
            if idx % 2 == 1:
                features.append(x)  # each SSD feature

        return tuple(features)

    @staticmethod
    def _vgg(
        vgg_config: List[Union[str, int]],
        in_channels: Union[str, int],
        batch_norm: bool,
    ) -> Iterable[nn.Module]:
        """Prepare VGG backbone."""
        for value in vgg_config:
            if value == "M":
                # standard max-pool
                yield nn.MaxPool2d(kernel_size=2, stride=2)
            elif value == "C":
                # max-pool with ceil-mode
                yield nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            else:
                # conv2d layer with batch-norm (optional) and ReLU
                yield nn.Conv2d(in_channels, value, kernel_size=3, padding=1)
                if batch_norm:
                    yield nn.BatchNorm2d(value)
                yield nn.ReLU(inplace=True)
                in_channels = value
        # common max-pool
        yield nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # next conv2d with ReLU
        yield nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        yield nn.ReLU(inplace=True)
        # final conv2d with ReLU
        yield nn.Conv2d(1024, 1024, kernel_size=1)
        yield nn.ReLU(inplace=True)

    @staticmethod
    def _vgg_extras(
        vgg_extras_config: List[Union[str, int]], in_channels: Union[str, int],
    ) -> Iterable[nn.Module]:
        """Add extra layers for SSD feature scaling."""
        kernel_size_flag = False
        for idx, value in enumerate(vgg_extras_config):
            if in_channels != "S":
                if value == "S":
                    # conv2d with stride
                    yield nn.Conv2d(
                        in_channels,
                        vgg_extras_config[idx + 1],
                        kernel_size=(1, 3)[kernel_size_flag],
                        stride=2,
                        padding=1,
                    )
                else:
                    # standard conv2d
                    yield nn.Conv2d(
                        in_channels, value, kernel_size=(1, 3)[kernel_size_flag]
                    )
                kernel_size_flag = not kernel_size_flag
            in_channels = value


class VGG300(VGG):
    """VGG300 backbone module."""

    def __init__(self, config: CfgNode):
        vgg300_config: List[Union[str, int]] = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "C",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
        ]
        vgg300_extras_config: List[Union[str, int]] = [
            256,
            "S",
            512,
            128,
            "S",
            256,
            128,
            256,
            128,
            256,
        ]
        vgg300_out_channels = [512, 1024, 512, 256, 256, 256]
        super().__init__(
            config=config,
            vgg_config=vgg300_config,
            vgg_extras_config=vgg300_extras_config,
            out_channels=vgg300_out_channels,
        )

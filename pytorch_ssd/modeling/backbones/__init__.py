from typing import Dict, Type

from pytorch_ssd.modeling.backbones.base import BaseBackbone
from pytorch_ssd.modeling.backbones.mobilenetv2 import MobileNetV2
from pytorch_ssd.modeling.backbones.vgg import (
    VGG300,
    VGG300BN,
    VGG512,
    VGG512BN,
    VGGLite,
    VGGLiteBN,
)

backbones: Dict[str, Type[BaseBackbone]] = {
    "VGGLite": VGGLite,
    "VGGLiteBN": VGGLiteBN,
    "VGG300": VGG300,
    "VGG300BN": VGG300BN,
    "VGG512": VGG512,
    "VGG512BN": VGG512BN,
    "mobilenetv2": MobileNetV2,
}

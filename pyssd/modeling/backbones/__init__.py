from typing import Dict, Type

import torch.nn as nn

from pyssd.modeling.backbones.mobilenetv2 import MobileNetV2
from pyssd.modeling.backbones.vgg import (
    VGG300,
    VGG300BN,
    VGG512,
    VGG512BN,
    VGGLite,
    VGGLiteBN,
)

backbones: Dict[str, Type[nn.Module]] = {
    "VGGLite": VGGLite,
    "VGGLiteBN": VGGLiteBN,
    "VGG300": VGG300,
    "VGG300BN": VGG300BN,
    "VGG512": VGG512,
    "VGG512BN": VGG512BN,
    "mobilenetv2": MobileNetV2,
}

from typing import Dict, Type

import torch.nn as nn

from pyssd.modeling.backbones.vgg import VGG300, VGG512, VGGLite

backbones: Dict[str, Type[nn.Module]] = {
    "VGG300": VGG300,
    "VGG512": VGG512,
    "VGGLite": VGGLite,
}

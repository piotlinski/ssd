from typing import Dict, Type

import torch.nn as nn

from pyssd.modeling.backbones.vgg import VGG11, VGG16

backbones: Dict[str, Type[nn.Module]] = {
    "VGG11": VGG11,
    "VGG16": VGG16,
}

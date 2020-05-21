from typing import Dict

import torch.nn as nn

from ssd.modeling.backbones.vgg import VGG300

backbones: Dict[str, nn.Module] = {"VGG300": VGG300}

from typing import Dict

import torch.nn as nn

from ssd.modeling.box_predictors.ssd import SSDBoxPredictor

box_predictors: Dict[str, nn.Module] = {"SSD": SSDBoxPredictor}
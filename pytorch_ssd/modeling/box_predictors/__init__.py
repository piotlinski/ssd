from typing import Dict, Type

from pytorch_ssd.modeling.box_predictors.ssd import BaseBoxPredictor, SSDBoxPredictor

box_predictors: Dict[str, Type[BaseBoxPredictor]] = {"SSD": SSDBoxPredictor}

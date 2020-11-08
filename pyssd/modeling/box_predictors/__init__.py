from typing import Dict, Type

from pyssd.modeling.box_predictors.ssd import BaseBoxPredictor, SSDBoxPredictor

box_predictors: Dict[str, Type[BaseBoxPredictor]] = {"SSD": SSDBoxPredictor}

"""SSD model."""
import torch.nn as nn
from yacs.config import CfgNode

from ssd.modeling.backbones import backbones
from ssd.modeling.box_predictors import box_predictors


class SSD(nn.Module):
    """SSD Detector class."""

    def __init__(self, config: CfgNode):
        super().__init__()
        self.config = config
        backbone = backbones[config.MODEL.BACKBONE]
        self.backbone = backbone(config)
        predictor = box_predictors[config.MODEL.BOX_PREDICTOR]
        self.predictor = predictor(
            config,
            backbone_out_channels=self.backbone.out_channels,
            boxes_per_location=config.DATA.PRIOR.BOXES_PER_LOC,
        )

    def forward(self, images):
        features = self.backbone(images)
        predictions = self.predictor(features)
        return predictions

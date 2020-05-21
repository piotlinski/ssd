"""SSD config."""
import logging
from pathlib import Path
from typing import Optional

from yacs.config import CfgNode

from ssd.modeling.backbones import backbones
from ssd.modeling.box_predictors import box_predictors

_C = CfgNode()

# data config
_C.DATA = CfgNode()
_C.DATA.CHANNELS = 3
_C.DATA.N_CLASSES = 10

# model config
_C.MODEL = CfgNode()
_C.MODEL.BATCH_NORM = True
_C.MODEL.PRETRAINED_URL = (
    "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
)
_C.MODEL.BACKBONE = "VGG300"
_C.MODEL.BOX_PREDICTOR = "SSD"


logger = logging.getLogger(__name__)


def verify_config(config: CfgNode):
    """Verify if chosen options are correct."""
    if config.MODEL.BACKBONE not in backbones:
        raise NameError("Backbone %s is not available", config.MODEL.BACKBONE)
    if config.MODEL.BOX_PREDICTOR not in box_predictors:
        raise NameError("Box predictor %s is not available", config.MODEL.BOX_PREDICTOR)


def get_config(config_file: Optional[Path] = None, **kwargs):
    """Get yacs config with default values."""
    config = _C.clone()
    if config_file is not None:
        if config_file.exists():
            config.merge_from_file(str(config_file))
        else:
            logger.warning("File %s does not exist.", str(config_file))
    config.update(**kwargs)
    config.freeze()
    verify_config(config)
    return config

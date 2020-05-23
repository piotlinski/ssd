"""SSD config."""
import logging
from pathlib import Path
from typing import Optional

from yacs.config import CfgNode

from ssd.data.datasets import datasets
from ssd.modeling.backbones import backbones
from ssd.modeling.box_predictors import box_predictors

_C = CfgNode()

# data config
_C.DATA = CfgNode()
_C.DATA.DATASET = "MultiscaleMNIST"
_C.DATA.DIR = "data"
_C.DATA.CHANNELS = 3
_C.DATA.SHAPE = (300, 300)
_C.DATA.N_CLASSES = 10
_C.DATA.PIXEL_MEAN = (0.0,)
_C.DATA.PIXEL_STD = (1.0,)
# data prior config
_C.DATA.PRIOR = CfgNode()
_C.DATA.PRIOR.BOXES_PER_LOC = (4, 6, 6, 6, 4, 4)
_C.DATA.PRIOR.FEATURE_MAPS = (38, 19, 10, 5, 3, 1)
_C.DATA.PRIOR.MIN_SIZES = (30, 60, 111, 162, 213, 264)
_C.DATA.PRIOR.MAX_SIZES = (60, 111, 162, 213, 264, 315)
_C.DATA.PRIOR.STRIDES = (8, 16, 32, 64, 100, 300)
_C.DATA.PRIOR.ASPECT_RATIOS = (
    (2,),
    (2, 3),
    (2, 3),
    (2, 3),
    (2,),
    (2,),
)
_C.DATA.PRIOR.CLIP = True

# model config
_C.MODEL = CfgNode()
_C.MODEL.BATCH_NORM = True
_C.MODEL.PRETRAINED_URL = (
    "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
)
_C.MODEL.BACKBONE = "VGG300"
_C.MODEL.BOX_PREDICTOR = "SSD"
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2
_C.MODEL.CONFIDENCE_THRESHOLD = 0.01
_C.MODEL.NMS_THRESHOLD = 0.45
_C.MODEL.MAX_PER_IMAGE = 100
_C.MODEL.IOU_THRESHOLD = 0.5

# runner config
_C.RUNNER = CfgNode()
_C.RUNNER.DEVICE = "gpu"
_C.RUNNER.BATCH_SIZE = 30
_C.RUNNER.LR = 1e-3
_C.RUNNER.NUM_WORKERS = 8
_C.RUNNER.PIN_MEMORY = True


logger = logging.getLogger(__name__)


def verify_config(config: CfgNode):
    """Verify if chosen options are correct."""
    if config.MODEL.BACKBONE not in backbones:
        raise NameError("Backbone %s is not available", config.MODEL.BACKBONE)
    if config.MODEL.BOX_PREDICTOR not in box_predictors:
        raise NameError("Box predictor %s is not available", config.MODEL.BOX_PREDICTOR)
    if config.DATA.DATASET not in datasets:
        raise NameError("Dataset %s is not available", config.DATA.DATASET)
    prior_len = len(config.DATA.PRIOR.BOXES_PER_LOC)
    if any(
        len(prior_config_tuple) != prior_len
        for prior_config_tuple in [
            config.DATA.PRIOR.BOXES_PER_LOC,
            config.DATA.PRIOR.FEATURE_MAPS,
            config.DATA.PRIOR.MIN_SIZES,
            config.DATA.PRIOR.MAX_SIZES,
            config.DATA.PRIOR.STRIDES,
            config.DATA.PRIOR.ASPECT_RATIOS,
        ]
    ):
        raise ValueError("Prior config is incorrect")


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

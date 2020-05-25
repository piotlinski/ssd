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
_C.DATA.CHANNELS = 1
_C.DATA.SHAPE = (300, 300)
_C.DATA.N_CLASSES = 11
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
# "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
_C.MODEL.PRETRAINED_URL = ""
_C.MODEL.BACKBONE = "VGG300"
_C.MODEL.BOX_PREDICTOR = "SSD"
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2
_C.MODEL.CONFIDENCE_THRESHOLD = 0.01
_C.MODEL.NMS_THRESHOLD = 0.45
_C.MODEL.MAX_PER_IMAGE = 100
_C.MODEL.IOU_THRESHOLD = 0.5
_C.MODEL.NEGATIVE_POSITIVE_RATIO = 3

# runner config
_C.RUNNER = CfgNode()
_C.RUNNER.DEVICE = "cuda"
_C.RUNNER.EPOCHS = 100
_C.RUNNER.BATCH_SIZE = 16
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


def get_config(config_file: Optional[str] = None, **kwargs) -> CfgNode:
    """Get yacs config with default values."""
    config = _C.clone()
    if config_file is not None:
        config_path = Path(config_file)
        if config_path.exists():
            config.merge_from_file(config_file)
        else:
            logger.warning("File %s does not exist.", config_file)
    config.update(**kwargs)
    config.freeze()
    verify_config(config)
    return config

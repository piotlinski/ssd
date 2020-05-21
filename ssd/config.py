"""SSD config."""
import logging
from pathlib import Path
from typing import Optional

from yacs.config import CfgNode

_C = CfgNode()

logger = logging.getLogger(__name__)


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
    return config

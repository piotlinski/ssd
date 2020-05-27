import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import torch
import torch.nn as nn
from torch.hub import HASH_REGEX, download_url_to_file
from yacs.config import CfgNode

logger = logging.getLogger(__name__)


def cache_url(
    url: str, model_dir: Optional[Path] = None, progress: bool = True
) -> Path:
    """ Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    The default value of `model_dir` is ``$TORCH_HOME/models`` where
    ``$TORCH_HOME`` defaults to ``~/.torch``. The default directory can be
    overridden with the ``$TORCH_MODEL_ZOO`` environment variable.
    Example:
        >>> cached_file = cache_url(
        >>>     'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'
        >>> )

    :param url: URL of object to download
    :param model_dir: directory in which to save the object
    :param progress: display a progressbar
    :return: Path to downloaded object
    """
    if model_dir is None:
        model_dir = Path(".").joinpath("models")
    if not model_dir.exists():
        model_dir.mkdir()

    parts = urlparse(url)
    filename = Path(parts.path).name

    if filename == "model_final.pkl":
        # workaround:
        # as pre-trained Caffe2 models from Detectron have all the same filename
        # so make the full path the filename by replacing / with _
        filename = parts.path.replace("/", "_")

    cached_file = model_dir.joinpath(filename)

    if not cached_file.exists():
        logger.info("Downloading: '%s' to %s.", url, str(cached_file))
        hash_prefix = HASH_REGEX.search(filename)
        if hash_prefix is not None:
            hash_prefix = hash_prefix.group(1)
            # workaround:
            # Caffe2 models don't have a hash, but follow the R-50 convention,
            # which matches the hash PyTorch uses. So we skip the hash matching
            # if the hash_prefix is less than 6 characters
            if len(hash_prefix) < 6:
                hash_prefix = None
        download_url_to_file(
            url, cached_file, hash_prefix=hash_prefix, progress=progress
        )
    return cached_file


class CheckPointer:
    """Class to handle model checkpointing."""

    _LAST_CHECKPOINT_FILENAME = "LAST_CHECKPOINT.txt"

    def __init__(self, config: CfgNode, model: nn.Module):
        """
        :param config: SSD config
        :param model: model to be checkpointed
        """
        self.model = model
        self.save_dir = Path(config.MODEL.CHECKPOINT_DIR)
        self.last_checkpoint_file = self.save_dir.joinpath(
            self._LAST_CHECKPOINT_FILENAME
        )

    @property
    def last_checkpoint(self) -> Optional[Path]:
        """Get path to last checkpoint."""
        last_checkpoint_path = self.last_checkpoint_file.read_text()
        if last_checkpoint_path:
            return Path(last_checkpoint_path)
        else:
            return None

    def save(self, filename: str):
        """ Save model to checkpoint and tag it.

        :param filename: checkpoint name
        """
        save_file = self.save_dir.joinpath(f"{filename}.pth")
        torch.save(self.model.state_dict(), save_file)
        logger.info(" CHECKPOINT | saved model checkpoint to %s" % save_file)
        self.last_checkpoint_file.write_text(str(save_file))

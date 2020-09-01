import logging
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import torch
import torch.nn as nn
from torch.hub import HASH_REGEX, download_url_to_file
from yacs.config import CfgNode

logger = logging.getLogger(__name__)


def cache_url(url: str, pretrained_directory: str, progress: bool = True) -> Path:
    """ Loads the Torch serialized object at the given URL.
    If the object is already present in `pretrained_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    Example:
        >>> cached_file = cache_url(
        >>>     'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
        >>>     'assets/pretrained',
        >>> )

    :param url: URL of object to download
    :param pretrained_directory: directory in which to save the object
    :param progress: display a progressbar
    :return: Path to downloaded object
    """
    pretrained_dir = Path(pretrained_directory)
    if not pretrained_dir.exists():
        pretrained_dir.mkdir()

    parts = urlparse(url)
    filename = Path(parts.path).name

    if filename == "model_final.pkl":
        # workaround:
        # as pre-trained Caffe2 models from Detectron have all the same filename
        # so make the full path the filename by replacing / with _
        filename = parts.path.replace("/", "_")

    cached_file = pretrained_dir.joinpath(filename)

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
        self.config = config
        self.model = model
        self.checkpoint_dir = Path(
            f"{config.ASSETS_DIR}/"
            f"{config.MODEL.CHECKPOINT_DIR}/"
            f"{self.config.EXPERIMENT_NAME}_{self.config.CONFIG_STRING}"
        )
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.last_checkpoint_file = self.checkpoint_dir.joinpath(
            self._LAST_CHECKPOINT_FILENAME
        )
        self.last_checkpoint_file.touch(exist_ok=True)

    def store_config(self):
        """Save config to yml file."""
        with self.checkpoint_dir.joinpath("config.yml").open("w") as f:
            with redirect_stdout(f):
                print(self.config.dump())

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
        save_file = self.checkpoint_dir.joinpath(f"{filename}.pth")
        torch.save(self.model.state_dict(), save_file)
        self.last_checkpoint_file.write_text(str(save_file))

    def load(self, filename: Optional[str] = None):
        """ Load model checkpoint. If no name provided - use latest.

        :param filename: optional checkpoint file to use
        """
        if filename is None and self.last_checkpoint is None:
            logger.info(" CHECKPOINT | No checkpoint chosen")
        else:
            load_file = (
                self.checkpoint_dir.joinpath(filename)
                if filename is not None
                else self.last_checkpoint
            )

            if not load_file.exists():  # type: ignore
                logger.info(" CHECKPOINT | Checkpoint %s not found", load_file)
            else:
                checkpoint = torch.load(load_file, map_location="cpu")
                self.model.load_state_dict(checkpoint)
                logger.info(" CHECKPOINT | loaded model checkpoint from %s" % load_file)

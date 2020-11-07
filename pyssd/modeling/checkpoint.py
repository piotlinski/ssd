# TODO use other way of saving config
import logging
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import torch
from yacs.config import CfgNode

from pyssd.modeling.model import SSD

logger = logging.getLogger(__name__)


class CheckPointer:
    """Class to handle model checkpointing."""

    _LAST_CHECKPOINT_FILENAME = "LAST_CHECKPOINT.txt"

    def __init__(
        self, config: CfgNode, checkpoint_dir: str, model: SSD, dataset_name: str
    ):
        """
        :param checkpoint_dir: checkpointing directory
        :param model: model to be checkpointed
        """
        self.config = config
        self.model = model
        self.checkpoint_dir = Path(
            f"{checkpoint_dir}/"
            f"{dataset_name}_{model.backbone.__name__}-{model.predictor.__name__}"
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
        """Save model to checkpoint and tag it.

        :param filename: checkpoint name
        """
        save_file = self.checkpoint_dir.joinpath(f"{filename}.pth")
        torch.save(self.model.state_dict(), save_file)
        self.last_checkpoint_file.write_text(str(save_file))

    def load(self, filename: Optional[str] = None):
        """Load model checkpoint. If no name provided - use latest.

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

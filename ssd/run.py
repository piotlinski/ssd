"""SSD running utils."""
import logging

import torch
from yacs.config import CfgNode

from ssd.modeling.model import SSD

logger = logging.getLogger(__name__)


class Runner:
    """SSD runner."""

    def __init__(self, config: CfgNode):
        """
        :param config: configuration object
        """
        self.config = config
        self.device = self.set_device()
        self.model = SSD(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.RUNNER.LR)
        self.model.to(self.device)

    def set_device(self) -> torch.device:
        """Set runner device."""
        return torch.device(
            "cuda"
            if torch.cuda.is_available() and self.config.RUNNER.DEVICE == "cuda"
            else "cpu"
        )

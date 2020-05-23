"""SSD running utils."""
import logging
import time
from datetime import timedelta

import numpy as np
import torch
from yacs.config import CfgNode

from ssd.data.loaders import TestDataLoader, TrainDataLoader
from ssd.loss import MultiBoxLoss
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

        self.data_loader = TrainDataLoader(config)
        self.eval_data_loader = TestDataLoader(config)

        self.criterion = MultiBoxLoss(config.MODEL.NEGATIVE_POSITIVE_RATIO)

    def set_device(self) -> torch.device:
        """Set runner device."""
        return torch.device(
            "cuda"
            if torch.cuda.is_available() and self.config.RUNNER.DEVICE == "cuda"
            else "cpu"
        )

    def train(self):
        self.model.train()
        """Train the model."""
        loss = np.nan
        n_epochs = self.config.RUNNER.EPOCHS
        start_time = time.time()
        for epoch in range(n_epochs):
            epoch += 1
            epoch_start = time.time()
            for images, locations, labels in self.data_loader:
                images = images.to(self.device)
                locations = locations.to(self.device)
                labels = labels.to(self.device)

                cls_logits, bbox_pred = self.model(images)

                loss = self.criterion(
                    confidence=cls_logits,
                    predicted_locations=bbox_pred,
                    labels=labels,
                    gt_locations=locations,
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            epoch_time = time.time() - epoch_start
            eta = (n_epochs - epoch) * timedelta(seconds=epoch_time)
            logger.info(
                "epoch: %6d | lr: %.5f | loss: %10.3f | eta: %s",
                epoch,
                self.optimizer.param_groups[0]["lr"],
                loss,
                str(eta),
            )
        total_time = timedelta(seconds=time.time() - start_time)
        logger.info(
            "Finished. Total training time %s (%.3f s / epoch)",
            str(total_time),
            total_time.total_seconds() / n_epochs,
        )

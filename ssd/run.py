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
        self.model.to(self.device)

        self.criterion = MultiBoxLoss(config.MODEL.NEGATIVE_POSITIVE_RATIO)

    def set_device(self) -> torch.device:
        """Set runner device."""
        return torch.device(
            "cuda"
            if torch.cuda.is_available() and self.config.RUNNER.DEVICE == "cuda"
            else "cpu"
        )

    def train(self):
        """Train the model."""
        n_epochs = self.config.RUNNER.EPOCHS
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.RUNNER.LR)
        data_loader = TrainDataLoader(self.config)
        start_time = time.time()
        for epoch in range(n_epochs):
            losses = []
            self.model.train()
            epoch += 1
            epoch_start = time.time()
            for images, locations, labels in data_loader:
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            epoch_time = time.time() - epoch_start
            eta = (n_epochs - epoch) * timedelta(seconds=epoch_time)
            logger.info(
                "TRAIN | epoch: %6d | lr: %.5f | loss: %10.3f | eta: %s",
                epoch,
                optimizer.param_groups[0]["lr"],
                np.average(losses),
                str(eta),
            )
            self.eval()
        total_time = timedelta(seconds=time.time() - start_time)
        logger.info(
            "Training finished. Total training time %s (%.3f s / epoch)",
            str(total_time),
            total_time.total_seconds() / n_epochs,
        )

    def eval(self):
        """Evaluate the model."""
        self.model.eval()
        data_loader = TestDataLoader(self.config)
        losses = []
        for images, locations, labels in data_loader:
            with torch.no_grad():
                cls_logits, bbox_pred = self.model(images)
            loss = self.criterion(
                confidence=cls_logits,
                predicted_locations=bbox_pred,
                labels=labels,
                gt_locations=locations,
            )
            losses.append(loss.item())
        logger.info(" EVAL | loss: %10.3f", np.average(losses))

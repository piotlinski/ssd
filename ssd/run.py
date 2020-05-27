"""SSD running utils."""
import logging
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from yacs.config import CfgNode

from ssd.data.loaders import TestDataLoader, TrainDataLoader
from ssd.data.transforms import DataTransform
from ssd.loss import MultiBoxLoss
from ssd.modeling.checkpoint import CheckPointer
from ssd.modeling.model import SSD, process_model_prediction

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

        self.checkpointer = CheckPointer(config=config, model=self.model)
        self.checkpointer.load(
            config.MODEL.CHECKPOINT_NAME if config.MODEL.CHECKPOINT_NAME else None
        )

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
        self.checkpointer.store_config()
        n_epochs = self.config.RUNNER.EPOCHS
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.RUNNER.LR)
        data_loader = TrainDataLoader(self.config)
        start_time = time.time()
        logger.info("Starting training for %d epochs", n_epochs)
        for epoch in range(n_epochs):
            losses = []
            self.model.train()
            epoch += 1
            epoch_start = time.time()
            pbar = tqdm(data_loader)
            pbar.set_description("TRAIN | loss ---.---")
            for images, locations, labels in pbar:
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
                pbar.set_description(f"TRAIN | loss {loss.item():7.3f}")
            epoch_time = time.time() - epoch_start
            eta = (n_epochs - epoch) * timedelta(seconds=epoch_time)
            logger.info(
                " TRAIN | epoch: %4d | lr: %.5f | loss: %7.3f | eta: %s",
                epoch,
                optimizer.param_groups[0]["lr"],
                np.average(losses),
                str(eta),
            )
            self.eval()
            self.checkpointer.save(
                f"{self.config.MODEL.BOX_PREDICTOR}"
                f"-{self.config.MODEL.BACKBONE}"
                f"_{self.config.DATA.DATASET}"
                f"-{epoch:04d}"
            )
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
            images = images.to(self.device)
            locations = locations.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                cls_logits, bbox_pred = self.model(images)

                loss = self.criterion(
                    confidence=cls_logits,
                    predicted_locations=bbox_pred,
                    labels=labels,
                    gt_locations=locations,
                )
            losses.append(loss.item())
        logger.info("  EVAL | loss: %7.3f", np.average(losses))

    def predict(
        self, inputs: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """ Perform predictions on given inputs.

        :param inputs: batch of images
        :return: model prediction
        """
        self.model.eval()
        transform = DataTransform(self.config)
        with Pool(processes=self.config.RUNNER.NUM_WORKERS) as pool:
            transformed_inputs, *_ = zip(*pool.map(transform, inputs))
        stacked_inputs = torch.stack(transformed_inputs)
        stacked_inputs = stacked_inputs.to(self.device)
        with torch.no_grad():
            cls_logits, bbox_pred = self.model(stacked_inputs)
        detections = process_model_prediction(self.config, cls_logits, bbox_pred)
        return detections

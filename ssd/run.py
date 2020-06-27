"""SSD running utils."""
import logging
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm, trange
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

        self.model_description = (
            f"{self.config.MODEL.BOX_PREDICTOR}"
            f"-{self.config.MODEL.BACKBONE}"
            f"_{self.config.DATA.DATASET}"
        )

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
        self.model.train()
        n_epochs = self.config.RUNNER.EPOCHS
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.RUNNER.LR)
        data_loader = TrainDataLoader(self.config)

        global_step = 0
        log_step_losses = []
        log_step_loss = float("nan")
        epoch_loss = float("nan")
        loss_eval = float("nan")

        logger.info(
            "Starting training %s for %d epochs", self.model_description, n_epochs
        )

        with trange(
            n_epochs,
            desc="  TRAINING",
            unit="epoch",
            postfix=dict(step=global_step, loss=epoch_loss),
        ) as epoch_pbar:
            for epoch in epoch_pbar:
                epoch_losses = []
                epoch += 1

                with tqdm(
                    data_loader,
                    desc=f"epoch {epoch:4d}",
                    unit="step",
                    postfix=dict(loss=log_step_loss, eval_loss=loss_eval),
                ) as step_pbar:
                    for images, locations, labels in step_pbar:
                        global_step += 1
                        images = images.to(self.device)
                        locations = locations.to(self.device)
                        labels = labels.to(self.device)

                        cls_logits, bbox_pred = self.model(images)

                        regression_loss, classification_loss = self.criterion(
                            confidence=cls_logits,
                            predicted_locations=bbox_pred,
                            labels=labels,
                            gt_locations=locations,
                        )
                        loss = regression_loss + classification_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_losses.append(loss.item())
                        log_step_losses.append(loss.item())

                        if global_step % self.config.RUNNER.LOG_STEP == 0:
                            log_step_loss = np.round(np.average(log_step_losses), 3)
                            log_step_losses = []
                            epoch_loss = np.round(np.average(epoch_losses), 3)

                        if global_step % self.config.RUNNER.EVAL_STEP == 0:
                            loss_eval = np.round(self.eval(), 3)
                            self.model.train()

                        if global_step % self.config.RUNNER.CHECKPOINT_STEP == 0:
                            self.checkpointer.save(
                                f"{self.model_description}"
                                f"-{epoch:04d}"
                                f"-{global_step:05d}"
                            )

                        epoch_pbar.set_postfix(step=global_step, loss=epoch_loss)
                        step_pbar.set_postfix(loss=log_step_loss, loss_eval=loss_eval)

        logger.info("Training finished")

    def eval(self) -> float:
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

                regression_loss, classification_loss = self.criterion(
                    confidence=cls_logits,
                    predicted_locations=bbox_pred,
                    labels=labels,
                    gt_locations=locations,
                )
                loss = regression_loss + classification_loss
            losses.append(loss.item())
        return np.average(losses)

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

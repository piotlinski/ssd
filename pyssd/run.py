"""SSD running utils."""
import logging
from multiprocessing import Pool
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from yacs.config import CfgNode

from pyssd.data.loaders import EvalDataLoader, TrainDataLoader
from pyssd.data.transforms import DataTransform
from pyssd.loss import MultiBoxLoss
from pyssd.modeling.checkpoint import CheckPointer
from pyssd.modeling.model import SSD, process_model_prediction
from pyssd.visualize import plot_images_from_batch

logger = logging.getLogger(__name__)


class PlateauWarmUpLRScheduler(ReduceLROnPlateau):
    """LR Scheduler with warm-up and reducing on plateau."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, *args, **kwargs):
        """
        :param optimizer: torch optimizer
        :param warmup_steps: number of steps on which LR will be increased
        """
        super().__init__(optimizer=optimizer, *args, **kwargs)
        self.warmup_params = [warmup_steps for _ in range(len(optimizer.param_groups))]
        self.last_step = -1
        self.target_lrs = [params["lr"] for params in optimizer.param_groups]

    @staticmethod
    def linear_warmup_factor(step: int, warmup_steps: int):
        """Get linear factor to multiply learning rate during warmup.

        :param step: current step
        :param warmup_steps: number of steps on which LR will be increased
        :return: factor to multiply learning rate during warmup
        """
        return min(1.0, (step + 1) / warmup_steps)

    def dampen(self, step: Optional[int] = None):
        """Dampen the learning rates.

        :param step: current step (optional)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        for group, target_lr, warmup_steps in zip(
            self.optimizer.param_groups, self.target_lrs, self.warmup_params
        ):
            factor = self.linear_warmup_factor(step, warmup_steps=warmup_steps)
            group["lr"] = factor * target_lr


class Runner:
    """SSD runner."""

    def __init__(self, config: CfgNode):
        """
        :param config: configuration object
        """
        self.config = config
        torch.set_num_threads(self.config.RUNNER.NUM_WORKERS)
        self.device = self.set_device()
        self.model = SSD(config)

        self.checkpointer = CheckPointer(config=config, model=self.model)
        self.checkpointer.load(
            config.MODEL.CHECKPOINT_NAME if config.MODEL.CHECKPOINT_NAME else None
        )

        self.tb_writer = None
        if config.RUNNER.USE_TENSORBOARD:
            self.tb_writer = SummaryWriter(
                log_dir=(
                    f"{self.config.ASSETS_DIR}/"
                    f"{self.config.RUNNER.TENSORBOARD_DIR}/"
                    f"{self.config.EXPERIMENT_NAME}_{self.config.CONFIG_STRING}"
                )
            )

        self.train_data_loader = TrainDataLoader(self.config)
        self.eval_data_loader = EvalDataLoader(self.config)

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
        lr_scheduler = PlateauWarmUpLRScheduler(
            optimizer=optimizer,
            patience=self.config.RUNNER.LR_REDUCE_PATIENCE,
            warmup_steps=self.config.RUNNER.LR_WARMUP_STEPS,
        )

        global_step = 0

        losses = []
        regression_losses = []
        classification_losses = []
        log_loss = float("nan")
        epoch_loss = float("nan")

        logger.info(
            "Starting training %s for %d epochs",
            f"{self.config.EXPERIMENT_NAME}_{self.config.CONFIG_STRING}",
            n_epochs,
        )

        with trange(
            n_epochs,
            desc="               TRAINING",
            unit="epoch",
            postfix=dict(step=global_step, loss=epoch_loss),
        ) as epoch_pbar:
            for epoch in epoch_pbar:
                epoch_losses = []
                epoch += 1
                visualize = epoch % self.config.RUNNER.VIS_EPOCHS == 0

                with tqdm(
                    self.train_data_loader,
                    desc=f"TRAIN |      epoch {epoch:4d}",
                    unit="step",
                    postfix=dict(loss=log_loss),
                ) as step_pbar:
                    for images, locations, labels in step_pbar:
                        global_step += 1
                        lr_scheduler.dampen()
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
                        losses.append(loss.item())

                        regression_losses.append(regression_loss.item())
                        classification_losses.append(classification_loss.item())

                        if global_step % self.config.RUNNER.LOG_STEP == 0:
                            log_loss = np.average(losses)
                            log_regression_loss = np.average(regression_losses)
                            log_classification_loss = np.average(classification_losses)
                            losses = []
                            regression_losses = []
                            classification_losses = []
                            epoch_loss = np.average(epoch_losses)

                            if self.tb_writer is not None:
                                self.tb_writer.add_scalar(
                                    tag="loss-total/train",
                                    scalar_value=log_loss,
                                    global_step=global_step,
                                )
                                self.tb_writer.add_scalar(
                                    tag="loss-regression/train",
                                    scalar_value=log_regression_loss,
                                    global_step=global_step,
                                )
                                self.tb_writer.add_scalar(
                                    tag="loss-classification/train",
                                    scalar_value=log_classification_loss,
                                    global_step=global_step,
                                )
                                self.tb_writer.add_scalar(
                                    tag="lr",
                                    scalar_value=optimizer.param_groups[0]["lr"],
                                    global_step=global_step,
                                )
                                if self.config.RUNNER.TRACK_MODEL_PARAMS:
                                    for name, params in self.model.named_parameters():
                                        module, *sub, param_type = name.split(".")
                                        self.tb_writer.add_histogram(
                                            tag=f"{param_type}"
                                            f"/{module}_{'-'.join(sub)}",
                                            values=params,
                                            global_step=global_step,
                                        )

                        epoch_pbar.set_postfix(step=global_step, loss=epoch_loss)
                        step_pbar.set_postfix(loss=log_loss)

                if self.tb_writer is not None and visualize:
                    self.tb_writer.add_figure(
                        tag="predictions/train",
                        figure=plot_images_from_batch(
                            self.config,
                            image_batch=images,
                            pred_cls_logits=cls_logits.detach(),
                            pred_bbox_pred=bbox_pred.detach(),
                            gt_labels=labels,
                            gt_bbox_pred=locations,
                        ),
                        global_step=global_step,
                    )
                validation_loss = self.eval(
                    global_step=global_step, visualize=visualize
                )
                self.checkpointer.save(
                    f"{self.config.EXPERIMENT_NAME}"
                    f"_{self.config.CONFIG_STRING}"
                    f"-{epoch:04d}"
                    f"-{global_step:05d}"
                )
                self.model.train()

                if (
                    epoch > self.config.RUNNER.LR_REDUCE_SKIP_EPOCHS
                    and validation_loss != float("nan")
                ):
                    lr_scheduler.step(validation_loss)

        if self.tb_writer is not None:
            self.tb_writer.add_graph(self.model, images)
        logger.info("Training finished")

    def eval(self, global_step: int = 0, visualize: bool = False) -> float:
        """Evaluate the model."""
        self.model.eval()
        regression_losses = []
        classification_losses = []
        losses = []
        with tqdm(
            self.eval_data_loader,
            desc=f"EVAL  | step {global_step:10d}",
            unit="step",
            postfix=dict(loss=float("nan")),
        ) as step_pbar:
            for images, locations, labels in step_pbar:
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
                regression_losses.append(regression_loss.item())
                classification_losses.append(classification_loss.item())
                losses.append(loss.item())

                step_pbar.set_postfix(loss=np.average(losses))

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(
                tag="loss-total/eval",
                scalar_value=np.average(losses),
                global_step=global_step,
            )
            self.tb_writer.add_scalar(
                tag="loss-regression/eval",
                scalar_value=np.average(regression_losses),
                global_step=global_step,
            )
            self.tb_writer.add_scalar(
                tag="loss-classification/eval",
                scalar_value=np.average(classification_losses),
                global_step=global_step,
            )
            if visualize:
                self.tb_writer.add_figure(
                    tag="predictions/eval",
                    figure=plot_images_from_batch(
                        self.config,
                        image_batch=images,
                        pred_cls_logits=cls_logits,
                        pred_bbox_pred=bbox_pred,
                        gt_labels=labels,
                        gt_bbox_pred=locations,
                    ),
                    global_step=global_step,
                )

        return np.average(losses)

    def predict(
        self, inputs: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Perform predictions on given inputs.

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

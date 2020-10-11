"""Dataset loaders."""
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from yacs.config import CfgNode

from pyssd.data.datasets import datasets
from pyssd.data.datasets.base import DataTransformType
from pyssd.data.transforms import DataTransform, SSDTargetTransform, TrainDataTransform


class DefaultDataLoader(DataLoader):
    """Default dataset loader."""

    def __init__(
        self,
        data_transform: DataTransformType,
        subset: str,
        batch_size: int,
        config: CfgNode,
    ):
        target_transform = SSDTargetTransform(config)
        dataset = datasets[config.DATA.DATASET](
            f"{config.ASSETS_DIR}/{config.DATA.DATASET_DIR}",
            data_transform=data_transform,
            target_transform=target_transform,
            subset=subset,
        )
        sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False
        )
        super().__init__(
            dataset,
            num_workers=config.RUNNER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            pin_memory=config.RUNNER.PIN_MEMORY,
        )
        self.CLASS_LABELS = (
            dataset.CLASS_LABELS
            if config.DATA.N_CLASSES != 2
            else [dataset.OBJECT_LABEL]
        )


class TrainDataLoader(DefaultDataLoader):
    """Train dataset loader."""

    def __init__(self, config: CfgNode):
        data_transform = TrainDataTransform(config)
        super().__init__(
            data_transform,
            subset="train",
            batch_size=config.RUNNER.BATCH_SIZE,
            config=config,
        )


class EvalDataLoader(DefaultDataLoader):
    """Eval dataset loader."""

    def __init__(self, config: CfgNode):
        data_transform = DataTransform(config)
        super().__init__(
            data_transform,
            subset="test",
            batch_size=config.RUNNER.BATCH_SIZE,
            config=config,
        )

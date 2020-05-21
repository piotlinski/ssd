"""Dataset loaders."""
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate
from yacs.config import CfgNode

from ssd.data.datasets import datasets
from ssd.data.datasets.base import DataTransformType
from ssd.data.transforms import DataTransform, SSDTargetTransform, TrainDataTransform


def collate(batch):
    """Collate data for SSD."""
    images, targets, labels = list(zip(*batch))
    images = default_collate(images)
    targets = default_collate(targets)
    labels = default_collate(labels)
    return images, targets, labels


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
            config.DATA.DIR,
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
            num_workers=config.TRAINER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            pin_memory=config.TRAINER.PIN_MEMORY,
            collate_fn=collate,
        )


class TrainDataLoader(DefaultDataLoader):
    """Train dataset loader."""

    def __init__(self, config: CfgNode):
        data_transform = TrainDataTransform(config)
        super().__init__(
            data_transform,
            subset="train",
            batch_size=config.TRAINER.BATCH_SIZE,
            config=config,
        )


class TestDataLoader(DefaultDataLoader):
    """Test dataset loader."""

    def __init__(self, config: CfgNode):
        data_transform = DataTransform(config)
        super().__init__(data_transform, subset="test", batch_size=1, config=config)

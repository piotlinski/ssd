"""Command Line Interface"""
import logging
from typing import Optional

import click as click

from ssd.config import get_config
from ssd.data.datasets import datasets
from ssd.run import Runner

logger = logging.getLogger(__name__)


@click.group(help="SSD", epilog="Copyright Trasee Sp. z o. o.")
@click.option("--config-file", default=None, help="path to config file", type=str)
@click.pass_context
def main(ctx: click.Context, config_file: Optional[str]):
    """Main group for subcommands."""
    ctx.ensure_object(dict)
    config = get_config(config_file=config_file)
    logging.basicConfig(level=logging.INFO)
    logger.info("Using config:\n %s" % config)
    ctx.obj["config"] = config
    ctx.obj["runner"] = Runner(config=ctx.obj["config"])


@main.command(help="Train model")
@click.pass_obj
def train(obj):
    """Train the model."""
    runner = obj["runner"]
    runner.train()


@main.command(help="Evaluate model")
@click.pass_obj
def evaluate(obj):
    """Evaluate the model."""
    runner = obj["runner"]
    runner.eval()


@main.group(help="Dataset tools")
@click.pass_obj
def dataset(obj):
    """Group for dataset tools."""
    dataset_name = obj["config"].DATA.DATASET
    obj["dataset"] = datasets[dataset_name](obj["config"].DATA.DIR, subset="train",)


@dataset.command(help="Get dataset statistics")
@click.pass_obj
def stats(obj):
    """Calculate dataset pixel mean and std."""
    pixel_mean, pixel_std = obj["dataset"].pixel_mean_std()
    click.echo("Dataset: %s" % obj["config"].DATA.DATASET)
    click.echo("Pixel mean: %s" % str(pixel_mean))
    click.echo("Pixel std: %s" % str(pixel_std))


if __name__ == "__main__":
    main()

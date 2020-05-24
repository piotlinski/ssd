"""Command Line Interface"""
import logging
from typing import Optional

import click as click

from ssd.config import get_config
from ssd.run import Runner

logger = logging.getLogger(__name__)


@click.group(help="SSD", epilog="Copyright Trasee Sp. z o. o.")
@click.option("--config-file", default=None, help="path to config file", type=str)
@click.pass_context
def main(ctx: click.Context, config_file: Optional[str]):
    """Main group for subcommands."""
    ctx.ensure_object(dict)
    config = get_config(config_file=config_file)
    logger.info("Using config: %s" % config)
    ctx.obj["config"] = config
    ctx.obj["runner"] = Runner(config=ctx.obj["config"])


@main.command(help="Train model")
@click.pass_obj
def train(obj):
    runner = obj["runner"]
    runner.train()


@main.command(help="Evaluate model")
@click.pass_obj
def evaluate(obj):
    runner = obj["runner"]
    runner.eval()


if __name__ == "__main__":
    main()

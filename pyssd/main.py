"""Main function for SSD training."""
import argparse
from argparse import ArgumentParser

import yaml
from pytorch_lightning import Trainer

from pyssd.modeling.model import SSD


class LoadFromYaml(argparse.Action):
    """Load arguments from YAML file."""

    def __call__(self, parser: ArgumentParser, namespace, values, option_string=None):
        with values as fp:
            for key, value in yaml.load(fp, Loader=yaml.Loader).items():
                setattr(namespace, key, value)


def main(hparams):
    """Main function that creates and trains SSD model."""
    hparams_dict = vars(hparams)
    model = SSD(**hparams_dict)
    trainer = Trainer.from_argparse_args(hparams)
    trainer.tune(model)
    trainer.fit(model)


def cli():
    """SSD CLI with argparse."""
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=open,
        action=LoadFromYaml,
        help="YAML file with arguments to use",
    )
    parser = SSD.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

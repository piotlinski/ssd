"""Main function for SSD training."""
from argparse import ArgumentParser

from pytorch_lightning import Trainer

from pyssd.modeling.model import SSD


def main(hparams):
    """Main function that creates and trains SSD model."""
    hparams_dict = vars(hparams)
    model = SSD(**hparams_dict)
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model)


def cli():
    """SSD CLI with argparse."""
    parser = ArgumentParser()
    parser = SSD.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

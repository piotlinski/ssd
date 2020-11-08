"""Main function for SSD training."""
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from pyssd.modeling.model import SSD


def main(hparams):
    """Main function that creates and trains SSD model."""
    if hparams.checkpoint is not None:
        model = SSD.load_from_checkpoint(
            checkpoint_path=hparams.checkpoint, hparams_file=hparams.hparams_file
        )
    else:
        model = SSD(**vars(hparams))
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="ckpt-{epoch:02d}-{val_loss:.2f}",
        save_top_k=hparams.n_checkpoints,
        mode="min",
    )
    trainer = Trainer.from_argparse_args(hparams, callbacks=[checkpoint_callback])
    trainer.tune(model)
    trainer.fit(model)


def cli():
    """SSD CLI with argparse."""
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to start training from",
    )
    parser.add_argument(
        "--hparams-file",
        type=str,
        default=None,
        help="Hparams file to load hyperparameters from",
    )
    parser = SSD.add_model_specific_args(parser)
    parser.add_argument(
        "--n-checkpoints", type=int, default=3, help="Number of top checkpoints to save"
    )
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

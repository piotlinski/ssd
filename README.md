# PyTorch-SSD

Single Shot Multi-Box Detector implementation in PyTorch.

This is the implementation used by [SSDIR](https://github.com/piotlinski/ssdir), a single-shot multi-object representation learning model.

## Development

Requirements:

- Install `poetry` (https://python-poetry.org/docs/#installation)
- Use `poetry` to handle requirements
  - Execute `poetry add <package_name>` to add new library
  - Execute `poetry install` to create virtualenv and install packages

## Training

To train the model use the `train.py` script. Activate the environment by running `poetry shell` and run `python train.py --help` to see all the available options.

See all the available datasets in [datasets directory](pytorch_ssd/data/datasets). To train on multiscale MNIST dataset generate the dataset using [multiscalemnist tool](https://github.com/piotlinski/multiscalemnist).

A trained model weights file can be used for training SSDIR model.

## Manual

Use `make` to run commands

- `make help` - show help
- `make test` - run tests
  - `args="--lf" make test` - run pytest tests with different arguments
- `make shell` - run poetry shell

"""Test WIDER FACE dataset."""
from inspect import cleandoc
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from pytorch_ssd.data.datasets.wider import WIDERFace


@patch("pytorch_ssd.data.datasets.wider.Path.open")
@patch("pytorch_ssd.data.datasets.wider.WIDERFace")
def test_wider_dataset_params(_wider_mock, _open_mock):
    """Test WIDERFace dataset params."""
    path = "."
    ds = WIDERFace(data_dir=path, subset="train")
    assert ds.data_dir == Path(path)
    assert ds.subset == "train"
    assert len(ds.CLASS_LABELS) == 1
    assert ds.OBJECT_LABEL


@patch("pytorch_ssd.data.datasets.wider.Path.open")
def test_parse_annotations(open_mock):
    """Test parsing WIDER FACE annotations."""
    data_dir = "test"
    subset = "train"
    filepath = "some_directory/some_image.jpg"
    open_mock.return_value.__enter__.return_value = StringIO(
        cleandoc(
            f"""
            {filepath}
            4
            69 359 50 36 1 0 0 0 0 1
            227 382 56 43 1 0 1 0 0 1
            296 305 44 26 1 0 0 0 0 1
            353 280 40 36 2 0 0 0 2 1
            """
        )
    )
    dataset = WIDERFace(data_dir=data_dir, subset=subset)
    test_annotation = dataset.annotations[
        Path(f"{data_dir}/WIDER_{subset}/images/{filepath}")
    ]
    assert torch.eq(
        test_annotation[0],
        torch.tensor(
            [
                [69, 359, 119, 395],
                [227, 382, 283, 425],
                [296, 305, 340, 331],
                [353, 280, 393, 316],
            ]
        ),
    ).all()
    assert torch.eq(test_annotation[1], 1).all()

"""Test CLEVR dataset."""
from pathlib import Path
from unittest.mock import patch

import pytest

from pytorch_ssd.data.datasets.clevr import CLEVR


@pytest.fixture
def sample_scene_and_annotation():
    """Return sample scene dict with processed annotation."""
    scene = {
        "image_index": 0,
        "objects": [
            {
                "color": "brown",
                "size": "large",
                "rotation": 178.92387258999463,
                "shape": "cylinder",
                "3d_coords": [
                    -1.4937210083007812,
                    -1.9936031103134155,
                    0.699999988079071,
                ],
                "material": "rubber",
                "pixel_coords": [119, 131, 10.801968574523926],
            },
            {
                "color": "gray",
                "size": "large",
                "rotation": 243.405459279722,
                "shape": "cube",
                "3d_coords": [1.555708646774292, -2.104736566543579, 0.699999988079071],
                "material": "rubber",
                "pixel_coords": [198, 190, 8.60103988647461],
            },
            {
                "color": "green",
                "size": "small",
                "rotation": 230.45235024165092,
                "shape": "cylinder",
                "3d_coords": [
                    -2.342184543609619,
                    -0.5205014944076538,
                    0.3499999940395355,
                ],
                "material": "rubber",
                "pixel_coords": [161, 118, 12.372727394104004],
            },
            {
                "color": "purple",
                "size": "large",
                "rotation": 31.654351858799153,
                "shape": "sphere",
                "3d_coords": [
                    -0.8073106408119202,
                    1.914123773574829,
                    0.699999988079071,
                ],
                "material": "metal",
                "pixel_coords": [282, 100, 12.495001792907715],
            },
            {
                "color": "gray",
                "size": "small",
                "rotation": 42.183287560575,
                "shape": "cube",
                "3d_coords": [
                    2.6763813495635986,
                    0.03453871235251427,
                    0.3499999940395355,
                ],
                "material": "metal",
                "pixel_coords": [337, 195, 9.161211967468262],
            },
        ],
        "relationships": {
            "right": [[1, 2, 3, 4], [3, 4], [1, 3, 4], [4], []],
            "behind": [[2, 3], [0, 2, 3, 4], [3], [], [0, 2, 3]],
            "front": [[1, 4], [], [0, 1, 4], [0, 1, 2, 4], [1]],
            "left": [[], [0, 2], [0], [0, 1, 2], [0, 1, 2, 3]],
        },
        "image_filename": "CLEVR_val_000000.png",
        "split": "val",
        "directions": {
            "right": [0.6563112735748291, 0.7544902563095093, -0.0],
            "behind": [-0.754490315914154, 0.6563112735748291, 0.0],
            "above": [0.0, 0.0, 1.0],
            "below": [-0.0, -0.0, -1.0],
            "left": [-0.6563112735748291, -0.7544902563095093, 0.0],
            "front": [0.754490315914154, -0.6563112735748291, -0.0],
        },
    }
    annotation = {
        "x_min": [82.711, 142.634, 144.406, 247.143, 307.474],
        "y_min": [81.902, 134.634, 92.955, 65.143, 165.474],
        "x_max": [155.289, 253.366, 177.594, 316.857, 366.526],
        "y_max": [184.486, 245.366, 144.103, 134.857, 224.526],
        "class": [27, 1, 69, 35, 52],
    }
    return scene, annotation


@patch("pytorch_ssd.data.datasets.clevr.zipfile.ZipFile")
@patch("pytorch_ssd.data.datasets.clevr.subprocess.call")
@patch("pytorch_ssd.data.datasets.clevr.Path.mkdir")
def test_download_clevr(_mkdir_mock, call_mock, zipfile_mock):
    """Verify if CLEVR dataset can be downloaded."""
    CLEVR.download("test")
    call_mock.assert_called_with(
        f"curl {CLEVR.CLEVR_URL} -o test/{CLEVR.CLEVR_URL.split('/')[-1]}", shell=True
    )
    zipfile_mock.assert_called_with(f"test/{CLEVR.CLEVR_URL.split('/')[-1]}")


@patch("pytorch_ssd.data.datasets.clevr.json.load", return_value={"scenes": []})
@patch("pytorch_ssd.data.datasets.clevr.Path.open")
def test_clevr_dataset_params(_open_mock, _load_mock):
    """Test CLEVR dataset params."""
    path = "."
    ds = CLEVR(data_dir=path, subset="train")
    assert ds.data_dir == Path(path)
    assert ds.subset == "train"
    assert len(ds.CLASS_LABELS) == 96
    assert ds.OBJECT_LABEL


@patch("pytorch_ssd.data.datasets.clevr.json.load", return_value={"scenes": []})
@patch("pytorch_ssd.data.datasets.clevr.Path.open")
def test_clevr_extract_bbox_and_label(
    _open_mock, _load_mock, sample_scene_and_annotation
):
    """Verify if bbox and label is extracted from scene properly."""
    scene, exp_ann = sample_scene_and_annotation
    ds = CLEVR(data_dir=".", subset="train")
    ann = ds.extract_bbox_and_label(scene)
    assert ann["class"] == exp_ann["class"]
    assert all(
        pytest.approx(x_min == exp)
        for x_min, exp in zip(ann["x_min"], exp_ann["x_min"])
    )
    assert all(
        pytest.approx(y_min == exp)
        for y_min, exp in zip(ann["y_min"], exp_ann["y_min"])
    )
    assert all(
        pytest.approx(x_max == exp)
        for x_max, exp in zip(ann["x_max"], exp_ann["x_max"])
    )
    assert all(
        pytest.approx(y_max == exp)
        for y_max, exp in zip(ann["y_max"], exp_ann["y_max"])
    )

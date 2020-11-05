"""Test CLEVR dataset."""
from pathlib import Path
from unittest.mock import patch

from pyssd.data.datasets.clevr import CLEVR


@patch("pyssd.data.datasets.clevr.zipfile.ZipFile")
@patch("pyssd.data.datasets.clevr.subprocess.call")
@patch("pyssd.data.datasets.clevr.Path.mkdir")
def test_download_clevr(_mkdir_mock, call_mock, zipfile_mock):
    """Verify if CLEVR dataset can be downloaded."""
    CLEVR.download("test")
    call_mock.assert_called_with(
        f"curl {CLEVR.CLEVR_URL} -o test/{CLEVR.CLEVR_URL.split('/')[-1]}", shell=True
    )
    zipfile_mock.assert_called_with(f"test/{CLEVR.CLEVR_URL.split('/')[-1]}")


@patch("pyssd.data.datasets.clevr.json.load", return_value={"a.jpg": [], "b.png": []})
@patch("pyssd.data.datasets.clevr.Path.open")
def test_clevr_dataset_params(_open_mock, _load_mock):
    """Test CLEVR dataset params."""
    path = "."
    ds = CLEVR(data_dir=path, subset="train")
    assert ds.data_dir == Path(path)
    assert ds.subset == "train"
    assert len(ds.CLASS_LABELS) == 96
    assert ds.OBJECT_LABEL

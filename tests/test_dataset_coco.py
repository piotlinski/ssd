"""Test COCO dataset."""
from pathlib import Path
from unittest.mock import call, patch

import pytest

from ssd.data.datasets.coco import COCODetection


@patch("ssd.data.datasets.coco.Path.unlink")
@patch("ssd.data.datasets.coco.zipfile.ZipFile")
@patch("ssd.data.datasets.coco.subprocess.call")
@patch("ssd.data.datasets.coco.Path.mkdir")
def test_download_coco(_mkdir_mock, call_mock, zipfile_mock, _unlink_mock):
    """Verify if COCO dataset can be downloaded."""
    COCODetection.download("test")
    call_mock.assert_has_calls(
        [
            call(f"curl {url} -o test/{url.split('/')[-1]}", shell=True)
            for _, url in COCODetection.COCO_URLS
        ],
        any_order=True,
    )
    zipfile_mock.assert_has_calls(
        [
            call(f"test/{url.split('/')[-1]}")
            for target_dir, url in COCODetection.COCO_URLS
        ],
        any_order=True,
    )


@patch("ssd.data.datasets.coco.COCO")
def test_coco_dataset_params(_coco_mock):
    """Test COCODetection dataset params."""
    path = "."
    ds = COCODetection(data_dir=path, subset="train")
    assert ds.data_dir == Path(path)
    assert ds.subset == "train"
    assert len(ds.CLASS_LABELS) == 80
    assert ds.OBJECT_LABEL


@pytest.mark.parametrize(
    "coco_bbox, expected_bbox",
    [
        ([0, 0, 50, 50], [0, 0, 50, 50]),
        ([15, 12, 45, 32], [15, 12, 60, 44]),
        [[127, 324, 14, 67], [127, 324, 141, 391]],
    ],
)
def test_coco_bbox_to_corner_bbox(coco_bbox, expected_bbox):
    """Verify converting coco bbox to x1y1x2y2 bbox."""
    assert COCODetection.coco_bbox_to_corner_bbox(coco_bbox) == expected_bbox

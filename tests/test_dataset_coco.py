"""Test COCO dataset."""
from unittest.mock import call, patch

from ssd.data.datasets.coco import COCO_URLS, download_coco


@patch("ssd.data.datasets.coco.Path.unlink")
@patch("ssd.data.datasets.coco.zipfile.ZipFile")
@patch("ssd.data.datasets.coco.subprocess.call")
@patch("ssd.data.datasets.coco.Path.mkdir")
def test_download_coco(_mkdir_mock, call_mock, zipfile_mock, _unlink_mock):
    """Verify if COCO dataset can be downloaded."""
    download_coco("test")
    call_mock.assert_has_calls(
        [
            call(f"curl {url} -o test/{url.split('/')[-1]}", shell=True)
            for url in COCO_URLS
        ],
        any_order=True,
    )
    zipfile_mock.assert_has_calls(
        [call(f"test/{url.split('/')[-1]}") for url in COCO_URLS], any_order=True
    )

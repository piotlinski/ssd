"""COCO dataset."""
import subprocess
import zipfile
from pathlib import Path

COCO_URLS = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]


def download_coco(path: str):
    """Download and extract COCO dataset """
    target_dir = Path(path)
    target_dir.mkdir(exist_ok=True)
    for url in COCO_URLS:
        filename = url.split("/")[-1]
        target_path = target_dir.joinpath(filename)
        cmd = f"curl {url} -o {str(target_path)}"
        subprocess.call(cmd, shell=True)
        with zipfile.ZipFile(str(target_path)) as zf:
            zf.extractall(path=path)
        target_path.unlink()

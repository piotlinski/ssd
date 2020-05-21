import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from torch.hub import HASH_REGEX, _download_url_to_file

logger = logging.getLogger(__name__)


def cache_url(
    url: str, model_dir: Optional[Path] = None, progress: bool = True
) -> Path:
    """ Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    The default value of `model_dir` is ``$TORCH_HOME/models`` where
    ``$TORCH_HOME`` defaults to ``~/.torch``. The default directory can be
    overridden with the ``$TORCH_MODEL_ZOO`` environment variable.
    Example:
        >>> cached_file = cache_url(
        >>>     'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'
        >>> )

    :param url: URL of object to download
    :param model_dir: directory in which to save the object
    :param progress: display a progressbar
    :return: Path to downloaded object
    """
    if model_dir is None:
        model_dir = Path(".").joinpath("models")
    if not model_dir.exists():
        model_dir.mkdir()

    parts = urlparse(url)
    filename = Path(parts.path).name

    if filename == "model_final.pkl":
        # workaround:
        # as pre-trained Caffe2 models from Detectron have all the same filename
        # so make the full path the filename by replacing / with _
        filename = parts.path.replace("/", "_")

    cached_file = model_dir.joinpath(filename)

    if not cached_file.exists():
        logger.info("Downloading: '%s' to %s.", url, str(cached_file))
        hash_prefix = HASH_REGEX.search(filename)
        if hash_prefix is not None:
            hash_prefix = hash_prefix.group(1)
            # workaround:
            # Caffe2 models don't have a hash, but follow the R-50 convention,
            # which matches the hash PyTorch uses. So we skip the hash matching
            # if the hash_prefix is less than 6 characters
            if len(hash_prefix) < 6:
                hash_prefix = None
        _download_url_to_file(
            url, cached_file, hash_prefix=hash_prefix, progress=progress
        )
    return cached_file

"""Test checkpointing the model."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from ssd.modeling.checkpoint import CheckPointer, cache_url


@pytest.fixture
def nn_module_mock() -> MagicMock:
    """Return mocked nn.Module for testing."""
    return MagicMock(nn.Module)


@patch("ssd.modeling.checkpoint.Path.exists")
@patch("ssd.modeling.checkpoint.Path.mkdir")
def test_default_folder_creation(mkdir_mock, exists_mock):
    """Verify if default folder is created."""
    exists_mock.side_effect = [False, True]
    cache_url("test")
    mkdir_mock.assert_called()


@patch("ssd.modeling.checkpoint.Path.exists", return_value=False)
@patch("ssd.modeling.checkpoint.Path.mkdir")
@patch("ssd.modeling.checkpoint.HASH_REGEX")
@patch("ssd.modeling.checkpoint.download_url_to_file")
def test_downloading_model(download_mock, hash_regex_mock, _mkdir_mock, _exists_mock):
    """Check if appropriate file is downloaded."""
    hash_regex_mock.search = MagicMock(return_value=None)
    cache_url("test")
    download_mock.assert_called_with(
        "test", Path("./models/test"), hash_prefix=None, progress=True
    )


@patch("ssd.modeling.checkpoint.Path.exists", return_value=False)
@patch("ssd.modeling.checkpoint.Path.mkdir")
@patch("ssd.modeling.checkpoint.download_url_to_file")
@patch("ssd.modeling.checkpoint.HASH_REGEX")
@patch("ssd.modeling.checkpoint.urlparse")
def test_handling_caffe_model(
    urlparse_mock, hash_regex_mock, download_mock, _mkdir_mock, _exists_mock,
):
    """Test if caffe models are handled correctly."""
    urlparse_mock.return_value.path = "model_final.pkl"
    search_mock = MagicMock()
    hash_regex_mock.search = search_mock
    search_mock.return_value.group.return_value = "12345"

    cache_url("test")
    search_mock.return_value.group.assert_called_with(1)
    download_mock.assert_called_with(
        "test", Path("./models/model_final.pkl"), hash_prefix=None, progress=True
    )


@patch("ssd.modeling.checkpoint.Path.read_text", return_value="test")
def test_last_checkpoint(read_text_mock, nn_module_mock, sample_config):
    """Test if last_checkpoint is generated correctly."""
    checkpointer = CheckPointer(config=sample_config, model=nn_module_mock)
    assert checkpointer.last_checkpoint == Path("test")
    read_text_mock.assert_called_once()


@patch("ssd.modeling.checkpoint.Path.write_text")
@patch("ssd.modeling.checkpoint.torch.save")
def test_saving(torch_save_mock, write_text_mock, nn_module_mock, sample_config):
    """Test if saving is conducted."""
    filename = "test"
    checkpointer = CheckPointer(config=sample_config, model=nn_module_mock)
    checkpointer.save(filename)
    file = Path(sample_config.MODEL.CHECKPOINT_DIR).joinpath(f"{filename}.pth")
    torch_save_mock.assert_called_with(nn_module_mock.state_dict(), file)
    write_text_mock.assert_called_with(str(file))

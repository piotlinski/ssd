"""Test checkpointing the model."""
# TODO adjust to changes
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from yacs.config import CfgNode

from pyssd.modeling.backbones import BaseBackbone
from pyssd.modeling.box_predictors import BaseBoxPredictor
from pyssd.modeling.checkpoint import CheckPointer
from pyssd.modeling.model import SSD


@pytest.fixture
def ssd_mock() -> MagicMock:
    """Return mocked SSD for testing."""
    ssd = MagicMock(SSD)
    ssd.backbone = MagicMock(BaseBackbone)
    ssd.backbone.__name__ = "backbone"
    ssd.predictor = MagicMock(BaseBoxPredictor)
    ssd.predictor.__name__ = "predictor"
    return ssd


@pytest.fixture
def config() -> CfgNode:
    """Return empty config."""
    return CfgNode()


@patch("pyssd.modeling.checkpoint.Path.touch")
@patch("pyssd.modeling.checkpoint.Path.mkdir")
@patch("pyssd.modeling.checkpoint.print")
@patch("pyssd.modeling.checkpoint.redirect_stdout")
@patch("pyssd.modeling.checkpoint.Path.open")
def test_storing_checkpoint_config(
    open_mock, redirect_mock, print_mock, _mkdir_mock, _touch_mock, ssd_mock, config
):
    """Verify if config is stored."""
    checkpointer = CheckPointer(
        checkpoint_dir="dir", model=ssd_mock, dataset_name="ds", config=config
    )
    checkpointer.store_config()
    redirect_mock.assert_called_with(open_mock.return_value.__enter__.return_value)
    print_mock.assert_called_with(config.dump())


@patch("pyssd.modeling.checkpoint.Path.touch")
@patch("pyssd.modeling.checkpoint.Path.mkdir")
@patch("pyssd.modeling.checkpoint.Path.read_text", return_value="test")
def test_last_checkpoint(read_text_mock, _mkdir_mock, _touch_mock, ssd_mock, config):
    """Test if last_checkpoint is generated correctly."""
    checkpointer = CheckPointer(
        checkpoint_dir="dir", model=ssd_mock, dataset_name="ds", config=config
    )
    assert checkpointer.last_checkpoint == Path("test")
    read_text_mock.assert_called_once()


@patch("pyssd.modeling.checkpoint.Path.touch")
@patch("pyssd.modeling.checkpoint.Path.mkdir")
@patch("pyssd.modeling.checkpoint.Path.read_text", return_value="")
def test_empty_last_checkpoint(
    read_text_mock, _mkdir_mock, _touch_mock, ssd_mock, config
):
    """Test if None returned for no last checkpoint."""
    checkpointer = CheckPointer(
        checkpoint_dir="dir", model=ssd_mock, dataset_name="ds", config=config
    )
    assert checkpointer.last_checkpoint is None
    read_text_mock.assert_called_once()


@patch("pyssd.modeling.checkpoint.Path.touch")
@patch("pyssd.modeling.checkpoint.Path.mkdir")
@patch("pyssd.modeling.checkpoint.Path.write_text")
@patch("pyssd.modeling.checkpoint.torch.save")
def test_saving(
    torch_save_mock, write_text_mock, _mkdir_mock, _touch_mock, ssd_mock, config
):
    """Test if saving is conducted."""
    checkpoint_dir = "dir"
    filename = "test"
    dataset_name = "ds"
    checkpointer = CheckPointer(
        checkpoint_dir=checkpoint_dir,
        model=ssd_mock,
        dataset_name=dataset_name,
        config=config,
    )
    checkpointer.save(filename)
    file = Path(
        f"{checkpoint_dir}"
        f"/{dataset_name}_{ssd_mock.backbone.__name__}-{ssd_mock.predictor.__name__}"
        f"/{filename}.pth"
    )
    torch_save_mock.assert_called_with(ssd_mock.state_dict(), file)
    write_text_mock.assert_called_with(str(file))


@patch("pyssd.modeling.checkpoint.Path.touch")
@patch("pyssd.modeling.checkpoint.Path.mkdir")
@patch("pyssd.modeling.checkpoint.Path.exists", return_value=True)
@patch("pyssd.modeling.checkpoint.Path.read_text", return_value="latest")
@patch("pyssd.modeling.checkpoint.torch.load")
def test_loading(
    torch_load_mock,
    _read_text_mock,
    _exists_mock,
    _mkdir_mock,
    _touch_mock,
    ssd_mock,
    config,
):
    """Test if loading from given checkpoint is conducted."""
    checkpointer = CheckPointer(
        checkpoint_dir="dir", model=ssd_mock, dataset_name="ds", config=config
    )
    checkpointer.load("test")
    path = checkpointer.checkpoint_dir.joinpath("test")
    torch_load_mock.assert_called_with(path, map_location="cpu")
    ssd_mock.load_state_dict.assert_called_once()


@patch("pyssd.modeling.checkpoint.Path.touch")
@patch("pyssd.modeling.checkpoint.Path.mkdir")
@patch("pyssd.modeling.checkpoint.Path.exists", return_value=True)
@patch("pyssd.modeling.checkpoint.Path.read_text", return_value="latest")
@patch("pyssd.modeling.checkpoint.torch.load")
def test_loading_last_checkpoint(
    torch_load_mock,
    _read_text_mock,
    _exists_mock,
    _mkdir_mock,
    _touch_mock,
    ssd_mock,
    config,
):
    """Verify if latest checkpoint is used when no checkpoint name provided."""
    checkpointer = CheckPointer(
        checkpoint_dir="dir", model=ssd_mock, dataset_name="ds", config=config
    )
    checkpointer.load()
    torch_load_mock.assert_called_with(checkpointer.last_checkpoint, map_location="cpu")
    ssd_mock.load_state_dict.assert_called_once()


@patch("pyssd.modeling.checkpoint.Path.touch")
@patch("pyssd.modeling.checkpoint.Path.mkdir")
@patch("pyssd.modeling.checkpoint.Path.exists", return_value=False)
@patch("pyssd.modeling.checkpoint.Path.read_text", return_value="latest")
@patch("pyssd.modeling.checkpoint.torch.load")
def test_loading_no_file(
    torch_load_mock,
    _read_text_mock,
    _exists_mock,
    _mkdir_mock,
    _touch_mock,
    ssd_mock,
    config,
):
    """Check if nothing is loaded when given file does not exist."""
    checkpointer = CheckPointer(
        checkpoint_dir="dir", model=ssd_mock, dataset_name="ds", config=config
    )
    checkpointer.load("test")
    torch_load_mock.assert_not_called()
    ssd_mock.load_state_dict.assert_not_called()


@patch("pyssd.modeling.checkpoint.Path.touch")
@patch("pyssd.modeling.checkpoint.Path.mkdir")
@patch("pyssd.modeling.checkpoint.Path.exists", return_value=False)
@patch("pyssd.modeling.checkpoint.Path.read_text", return_value="")
@patch("pyssd.modeling.checkpoint.torch.load")
def test_loading_no_checkpoint(
    torch_load_mock,
    _read_text_mock,
    _exists_mock,
    _mkdir_mock,
    _touch_mock,
    ssd_mock,
    config,
):
    """Check if nothing is loaded when given file does not exist."""
    checkpointer = CheckPointer(
        checkpoint_dir="dir", model=ssd_mock, dataset_name="ds", config=config
    )
    checkpointer.load()
    torch_load_mock.assert_not_called()
    ssd_mock.load_state_dict.assert_not_called()

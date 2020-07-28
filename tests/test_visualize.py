"""Test visualization."""
from unittest.mock import MagicMock, patch

import pytest

from ssd.visualize import plot_image


@patch("matplotlib.pyplot.Axes.add_patch")
@patch("matplotlib.pyplot.Axes.imshow")
def test_plot_image_only(imshow_mock, add_patch_mock, sample_image, sample_config):
    """Test if only image is plotted correctly."""
    plot_image(sample_config, sample_image)
    imshow_mock.assert_called()
    add_patch_mock.assert_not_called()


@patch("matplotlib.pyplot.Axes.add_patch")
@patch("matplotlib.pyplot.Axes.imshow")
def test_plot_image_with_bbox(
    imshow_mock, add_patch_mock, sample_image, sample_prediction, sample_config
):
    """Test if only image is plotted correctly."""
    cls_logits, bbox_pred = sample_prediction
    data_loader = MagicMock()
    data_loader.CLASS_LABELS = ["test"] * (sample_config.DATA.N_CLASSES - 1)
    plot_image(
        sample_config,
        sample_image,
        prediction=(cls_logits.squeeze(0), bbox_pred.squeeze()),
        data_loader=data_loader,
    )
    imshow_mock.assert_called()
    add_patch_mock.assert_called()


@patch("matplotlib.pyplot.Axes.imshow")
def test_error_when_no_data_loader(
    _imshow_mock, sample_image, sample_prediction, sample_config
):
    """Verify if error is raised when no data loader provided for plotting labels."""
    cls_logits, bbox_pred = sample_prediction
    with pytest.raises(AttributeError):
        plot_image(
            sample_config,
            sample_image,
            prediction=(cls_logits.squeeze(0), bbox_pred.squeeze()),
        )

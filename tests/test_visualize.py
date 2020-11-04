"""Test visualization."""
from unittest.mock import patch

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
    plot_image(
        sample_config,
        sample_image,
        prediction=(cls_logits.squeeze(0), bbox_pred.squeeze()),
    )
    imshow_mock.assert_called()
    add_patch_mock.assert_called()

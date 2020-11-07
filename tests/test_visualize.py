"""Test visualization."""
from unittest.mock import patch

from pyssd.modeling.model import SSD
from pyssd.visualize import plot_image


@patch("matplotlib.pyplot.Axes.add_patch")
@patch("matplotlib.pyplot.Axes.imshow")
def test_plot_image_only(imshow_mock, add_patch_mock, sample_image, ssd_params):
    """Test if only image is plotted correctly."""
    model = SSD(**ssd_params)
    plot_image(sample_image, model=model)
    imshow_mock.assert_called()
    add_patch_mock.assert_not_called()


@patch("matplotlib.pyplot.Axes.add_patch")
@patch("matplotlib.pyplot.Axes.imshow")
def test_plot_image_with_bbox(imshow_mock, add_patch_mock, sample_image, ssd_params):
    """Test if only image is plotted correctly."""
    model = SSD(**ssd_params)
    cls_logits, bbox_pred = model.predictor(model.backbone(sample_image.unsqueeze(0)))
    plot_image(
        sample_image,
        model=model,
        prediction=(cls_logits.squeeze(0), bbox_pred.squeeze()),
        confidence_threshold=0.0,
    )
    imshow_mock.assert_called()
    add_patch_mock.assert_called()

"""Test base backbone."""
from unittest.mock import patch

import torch.nn as nn

from ssd.modeling.backbones.base import BaseBackbone


@patch(
    "ssd.modeling.backbones.base.BaseBackbone._build_extras", return_value=nn.Module()
)
@patch(
    "ssd.modeling.backbones.base.BaseBackbone._build_backbone", return_value=nn.Module()
)
@patch("ssd.modeling.backbones.base.nn.Module.load_state_dict")
@patch("ssd.modeling.backbones.base.torch.load")
@patch("ssd.modeling.backbones.base.cache_url")
def test_downloading_pretrained(
    cache_url_mock,
    torch_load_mock,
    load_state_dict_mock,
    _build_backbone_mock,
    _build_extras_mock,
    sample_config,
):
    """Test if pretrained weights tools are called."""
    sample_config.MODEL.PRETRAINED_URL = "test"
    BaseBackbone(sample_config, [1])
    cache_url_mock.assert_called_with(
        "test", f"{sample_config.ASSETS_DIR}/{sample_config.MODEL.PRETRAINED_DIR}"
    )
    torch_load_mock.assert_called_with(cache_url_mock.return_value, map_location="cpu")
    load_state_dict_mock.assert_called_with(torch_load_mock.return_value)

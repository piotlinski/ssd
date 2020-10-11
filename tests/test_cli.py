"""Test command line interface."""
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from pyssd.cli import download, evaluate, stats, train
from pyssd.data.datasets import BaseDataset
from pyssd.run import Runner


@patch("pyssd.run.CheckPointer")
@patch("pyssd.run.SSD")
@pytest.mark.parametrize(
    "command, args, task_mock",
    [
        (train, [], "pyssd.cli.Runner.train"),
        (evaluate, [], "pyssd.cli.Runner.eval"),
        (stats, [], "pyssd.data.datasets.base.BaseDataset.pixel_mean_std"),
        (download, [], "pyssd.data.datasets.base.BaseDataset.download"),
    ],
)
def test_failed_commands_exit_code(
    _ssd_mock, _checkpointer_mock, command, args, task_mock, sample_config
):
    """Test if raising unhandled exception return exit code 1"""
    runner = CliRunner()
    exception = RuntimeError("Random error")

    ssd_runner = Runner(sample_config)
    dataset = BaseDataset

    with patch(task_mock, side_effect=exception):
        result = runner.invoke(
            command,
            args,
            obj={
                "config": sample_config,
                "runner": ssd_runner,
                "dataset": dataset,
            },
        )

    assert result.exit_code == 1
    assert result.exception == exception

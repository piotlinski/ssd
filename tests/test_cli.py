"""Test command line interface."""
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ssd.cli import evaluate, train
from ssd.run import Runner


@patch("ssd.run.SSD")
@pytest.mark.parametrize(
    "command, args, task_mock",
    [(train, [], "ssd.cli.Runner.train"), (evaluate, [], "ssd.cli.Runner.eval"),],
)
def test_failed_commands_exit_code(_ssd_mock, command, args, task_mock, sample_config):
    """Test if raising unhandled exception return exit code 1"""
    runner = CliRunner()
    exception = RuntimeError("Random error")

    ssd_runner = Runner(sample_config)

    with patch(task_mock, side_effect=exception):
        result = runner.invoke(
            command, args, obj={"config": sample_config, "runner": ssd_runner}
        )

    assert result.exit_code == 1
    assert result.exception == exception
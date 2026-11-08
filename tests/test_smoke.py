from typer.testing import CliRunner

import swarmguard
from swarmguard.cli import app


def test_version_attr():
    assert swarmguard.__version__


def test_cli_help():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "swarmguard" in result.output

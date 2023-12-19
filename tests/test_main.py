from click.testing import CliRunner
import toml

from whisperlab.__main__ import cli


META = toml.load("pyproject.toml")
VERSION = META["project"]["version"]


def test_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0


def test_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert VERSION in result.output


def test_transcribe():
    runner = CliRunner()
    result = runner.invoke(cli, ["transcribe", "tests/data/test.wav"])
    assert result.exit_code == 0
    assert "hello world" in result.output.lower()

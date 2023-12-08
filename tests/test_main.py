from click.testing import CliRunner


from whisperlab.__main__ import cli


def test_cli_entry():
    CliRunner.invoke(cli, ["--help"])

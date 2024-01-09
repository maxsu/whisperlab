from whisperlab import VERSION

import subprocess


def run_whisperlab(*args):
    """Run whisperlab and return the result."""

    result: subprocess.CompletedProcess = subprocess.run(
        ["whisperlab", *args],
        stdout=subprocess.PIPE,  # Capture stdout
    )
    # Transform the stdout bytes buffer to string
    result.output = result.stdout.decode()
    return result


def test_help():
    result = run_whisperlab("--help")
    assert result.returncode == 0


def test_version():
    result = run_whisperlab("--version")
    assert result.returncode == 0
    assert VERSION in result.output


def test_transcribe():
    audio_file = "audio/hello_world.mp3"
    result = run_whisperlab("transcribe", audio_file)
    assert result.returncode == 0
    assert "Hello world." in result.output

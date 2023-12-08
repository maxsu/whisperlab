"""
The main entry point for the CLI.

This module is responsible for:

1. Starting application services
2. Parsing & Validating CLI arguments
3. Calling the appropriate use case

Usage Examples:
whisperlab transcribe audio.wav
whisperlab transcribe audio.wav --model english
whisperlab transcribe audio.wav -m english
"""

import click

from whisperlab.run_whisper import (
    run_whisper,
    WhisperRequest,
    WhisperModels,
    DEFAULT_TRANSRIPTION_MODEL,
)
from whisperlab.logging import config_log


# Logging =====================================================================

config_log()


# Click Objects ===============================================================

# Validators
ExistingFile = click.Path(exists=True, dir_okay=False)

# CLI Commands ================================================================


# Base Group
@click.group()
def cli():
    pass


# Transcribe Command
@cli.command()
@click.argument("audio_file", type=ExistingFile)
@click.option(
    "-m",
    "--model",
    type=click.Choice(WhisperModels),
    default=DEFAULT_TRANSRIPTION_MODEL,
    help="The transcription model to use",
)
def transcribe(audio_file: str, model: str):
    """
    Transcribe an audio file.

    Args:
        audio_file (str): The audio file to transcribe
        model (str): The transcription model to use
    """
    request = WhisperRequest(audio_file, model=model)
    run_whisper(request)


# Run the CLI =================================================================

if __name__ == "__main__":
    cli()

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

import logging
from pathlib import Path

import click

from whisperlab import VERSION
from whisperlab.transcribe import (
    transcribe as transcription_use_case,
    TRANSCRIPTION_MODELS,
    DEFAULT_TRANSCRIPTION_MODEL,
    TranscribeTask,
)
import whisperlab.logging

# Logging =====================================================================


log = whisperlab.logging.config_log()

# Click Objects ===============================================================

# Validators
ExistingFile = click.Path(exists=True, dir_okay=False)

# CLI Commands ================================================================


# Base Group
@click.group()
@click.version_option(version=VERSION)
def cli():
    pass


# Transcribe Command
@cli.command()
@click.argument("audio_file", type=ExistingFile)
@click.option(
    "-m",
    "--model",
    type=click.Choice(TRANSCRIPTION_MODELS),
    default=DEFAULT_TRANSCRIPTION_MODEL,
    help="The transcription model to use",
)
def transcribe(audio_file: str, model: str):
    """
    Transcribe an audio file.

    Args:
        audio_file (str): The audio file to transcribe
        model (str): The transcription model to use
    """
    transcription_task = TranscribeTask(audio_file=Path(audio_file), model=model)
    transcription_use_case(transcription_task)


# Run the CLI =================================================================

if __name__ == "__main__":
    cli()

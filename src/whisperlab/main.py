"""
The main entry point for the CLI.

This module is responsible for:

1. Starting application services
2. Parsing & Validating CLI arguments
3. Calling the appropriate use case
"""

from click import Choice, option, argument, group, Path

from .Service import logger
from whisperlab.run_whisper import (
    run_whisper,
    WhisperRequest,
    WhisperModels,
    DEFAULT_WHISPER_MODEL,
)

# Click Validators
WriteableFile = Path(writable=True, dir_okay=False)
ExistingFile = Path(exists=True, dir_okay=False)


# CLI Commands ================================================================

# Shared options


@group()
@option("-v", "--verbose", help="Enable verbose output.")
@option("-s", "--structured", help="Enable JSON output.")
@option("-l", "--log-file", help="Log output to file.", type=WriteableFile)
def cli(verbose: bool, structured: bool, log_file: str):
    """
    Take care of options shared by multiple commands

    Args:
        verbose (bool): Enable verbose output
        structured (bool): Enable JSON output
        log_file (str): Log output to this file
    """
    # Set up the logger
    logger.config(verbose, structured, log_file)


# Transcribe command


@cli.command()
@argument(
    "audio_file",
    help="The audio file to transcribe",
    type=ExistingFile,
    required=True,
)
@option(
    "-m",
    "--model",
    help="The transcription model to use.",
    type=Choice(WhisperModels),
    default=DEFAULT_WHISPER_MODEL,
)
def transcribe(audio_file: str, model: str):
    """
    Transcribe an audio file.

    Args:
        audio_file (str): The audio file to transcribe
        model (str): The transcription model to use
    """
    request = WhisperRequest(audio_file, model=model)
    run_whisper(model, request)


# Run the CLI =================================================================

if __name__ == "__main__":
    cli()

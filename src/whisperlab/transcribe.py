"""
Whisper Runner Module

This module can transcribe an audio file using Whisper.
"""

import logging
from pathlib import Path

import whisper

from .tasks import TranscriptionTask


log = logging.getLogger("main")


# Globals =====================================================================

TRANSCRIPTION_MODELS = ["base"]

DEFAULT_TRANSCRIPTION_MODEL = TRANSCRIPTION_MODELS[0]

EMPTY_RESULT = {"text": ""}


# Validation ==================================================================


def EmptyFile(audio_file: Path):
    """
    Validate that the audio file is not empty

    Args:
        audio_file (Path): Path to the audio file to validate

    Returns:
        bool: True if the audio file is empty, False otherwise
    """
    return audio_file.stat().st_size == 0


# Use Case ====================================================================


def transcribe(
    task: TranscriptionTask,
):
    """
    Run trancription on an audio file.

    Args:
        task (TranscriptionTask): The trascription task to process.

    Effects:
        Logs the result.

    Returns:
        task (TranscriptionTask): The trascription task with the result.
    """

    # Check if the audio file is empty
    if EmptyFile(task.audio_file):
        # Return an empty result
        task.result = EMPTY_RESULT
        return task

    # Load and trim the audio file to 30 seconds
    audio = whisper.load_audio(str(task.audio_file))
    audio = whisper.pad_or_trim(audio)

    # Log the audio file
    log.info("Transcribing %s", task.audio_file)

    # Fetch the model
    model = whisper.load_model(task.model)

    # Transcribe the audio
    response = model.transcribe(audio, **task.args)

    # Log the result text
    log.info("Transcription:\n%s", response["text"])

    return response

"""
Whisper Runner Module

This module can transcribe an audio file using Whisper.
"""

from pathlib import Path
from pydantic import FilePath
import whisper

import whisperlab.logging
from .tasks import Task


log = whisperlab.logging.config_log()

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


# Models ======================================================================


class TranscribeTask(Task):
    """
    Whisper Request Model

    Args:
        audio_file (Path): Path to the audio file to transcribe
        args (dict): Arguments to pass to whisper

    Returns:
        dict: The whisper result
    """

    audio_file: FilePath
    args: dict = {}
    model: str = DEFAULT_TRANSCRIPTION_MODEL


# Use Case ====================================================================


def transcribe(
    task: TranscribeTask,
):
    """
    Run trancription on an audio file.

    Args:
        task (TranscribeTask): The trascription task to process.

    Effects:
        Logs the result.

    Returns:
        task (TranscribeTask): The trascription task with the result.
    """

    # Validate empty audio files
    if EmptyFile(task.audio_file):
        return EMPTY_RESULT

    # Load and trim the audio file to 30 seconds
    audio = whisper.load_audio(str(task.audio_file))
    audio = whisper.pad_or_trim(audio)

    # Log the audio file
    log.info("Transcribing %s", task.audio_file)

    # Fetch the model
    model = whisper.load_model(task.model)

    # Transcribe the audio
    result = model.transcribe(audio, fp16=False, **task.args)

    # Log the result text
    log.info("Transcription:\n%s", result["text"])

    return result

"""
Whisper Runner Module

This module can transcribe an audio file using Whisper.
"""

import logging

import whisper

from whisperlab import tasks


log = logging.getLogger("main")


# Config Objects ===============================================================


class WhisperModels:
    BASE = "base"


DEFAULT_WHISPER_MODEL = WhisperModels.BASE


# Use Case =====================================================================


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
    if task.audio_file.stat().st_size == 0:
        # Return an empty result
        task.result = {"text": ""}
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

"""
Whisper Runner Module

This module contains the run_whisper function which can transcribe an audio file.
"""

from pydantic import BaseModel, FilePath
import whisper

from whisperlab import logger

# Config

class WhisperModels:
    BASE = "base"

DEFAULT_WHISPER_MODEL = WhisperModels.BASE


# Request Model

class WhisperRequest(BaseModel):
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
    model: str = DEFAULT_WHISPER_MODEL


# Use Case

def run_whisper(
    request: WhisperRequest,
):
    """
    Run whisper on an audio file

    Args:
        request (WhisperRequest): The whisper request to process

    Effects:
        Logs the whisper result

    Returns:
        dict: The whisper result
    """

    # Load and trim the audio file to 30 seconds
    audio = whisper.load_audio(str(request.audio_file))
    audio = whisper.pad_or_trim(audio)

    # Log the audio file
    logger.info("Transcribing %s", request.audio_file)

    # Fetch the model
    model = whisper.load_model(request.model)

    # Transcribe the audio
    response = model.transcribe(audio, **request.args)

    # Log the result text
    logger.info("Transcription:\n%s", response["text"])

    return response
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, FilePath

from whisperlab.transcribe import DEFAULT_WHISPER_MODEL


class Task(BaseModel):
    id: str = uuid4()


# Models ===============================================================


class TransciptionTask(Task):
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

"""
Whisper Runner Module

This module can transcribe an audio file using Whisper.
"""

import logging
from pathlib import Path
import numpy as np
from pydantic import BaseModel, FilePath, root_validator, validator
import whisper

from whisperlab.time import time_ms, timestamp
from whisperlab.tasks import Task
from whisperlab.audio import (
    SAMPLES_PER_SECOND,
    Microphone,
    save_audio_segment,
)

log = logging.getLogger("main")


# Whisper =====================================================================

TRANSCRIPTION_MODELS = ["base"]
DEFAULT_TRANSCRIPTION_MODEL = TRANSCRIPTION_MODELS[0]


def get_model():
    _time = time_ms()
    model = whisper.load_model(DEFAULT_TRANSCRIPTION_MODEL)
    log.debug("Loaded model in %s ms", time_ms() - _time)
    return model


# Base Use Case ===============================================================


class TranscribeTask(Task):
    """
    Whisper Request Model

    Args:
        audio_file (Path): Path to the audio file to transcribe
        args (dict): Arguments to pass to whisper

    Returns:
        dict: The whisper result
    """

    # Audio source
    audio_file: str | FilePath = None
    samples: np.ndarray = None

    # Transcription options
    context: str = ""
    args: dict = {}
    model: str | whisper.Whisper = DEFAULT_TRANSCRIPTION_MODEL

    # Task status flags
    empty: bool = False

    # Validators ----------------------------------------------------------------

    @validator("audio_file", pre=True)
    def validate_audio_file(cls, audio_file):
        if isinstance(audio_file, str):
            audio_file = Path(audio_file)
        if not audio_file.exists():
            raise ValueError(f"Audio file does not exist: {audio_file}")
        return audio_file

    @root_validator(pre=True)
    def check_audio_file_or_samples(cls, values):
        audio_file, samples = values.get("audio_file"), values.get("samples")
        if audio_file is None and samples is None:
            raise ValueError("Please provide either 'audio_file' or 'samples'")
        if audio_file is not None and samples is not None:
            raise ValueError(
                "Please provide either 'audio_file' or 'samples', not both."
            )
        return values

    @root_validator(pre=True)
    def flag_empty_inputs(cls, values):
        audio_file, samples = values.get("audio_file"), values.get("samples")
        if audio_file is not None:
            if audio_file.stat().st_size == 0:
                values["empty"] = True
        else:
            if len(samples) == 0:
                values["empty"] = True
        return values

    # -------------------------------------------------------------------------

    def load_resources(self):
        # Load the model if necessary
        if isinstance(self.model, str):
            self.model = whisper.load_model(self.model)

        # Load the audio if necessary
        if self.samples is None:
            self.samples = whisper.load_audio(str(self.audio_file))

        # Trim or pad the audio to 30 seconds
        self.samples = whisper.pad_or_trim(self.samples)

    def dump(self):
        # We want to save the audio and the transcription
        if not self.completed:
            raise NotImplementedError("Trying to save an incomplete task.")

        save_audio_segment(
            self.samples,
            self.result["text"],
            Path("logs") / self.batch_name / f"sequence={self.sequence_num}",
        )


class TranscribeResult(BaseModel):
    text: str = ""


def transcribe(task: TranscribeTask) -> TranscribeResult:
    """
    Run trancription on an audio file.

    Args:
        task (TranscribeTask): The trascription task to process.

    Effects:
        Logs the result.

    Returns:
        task (TranscribeTask): The trascription task with the result.
    """

    # Handle empty task
    if task.empty:
        result = TranscribeResult()  # Empty result
        task.complete(result)
        return result

    # Load the model and audio
    task.load_resources()

    # Transcribe the audio
    result = TranscribeResult()
    result.text = task.model.transcribe(
        task.samples,
        fp16=False,
        **task.args,
    )["text"]
    task.complete(result)

    # Log the result text
    log.info("Transcription:\n%s", result.text)

    return result


# Real Time Use Case ==========================================================

WINDOW_SECONDS = 5
SAMPLES_PER_WINDOW = SAMPLES_PER_SECOND * WINDOW_SECONDS


class TranscribeRTTask(Task):
    microphone_class: type[Microphone] = Microphone
    model: str | whisper.Whisper = DEFAULT_TRANSCRIPTION_MODEL

    def load_resources(self):
        # Load the model if necessary
        if isinstance(self.model, str):
            self.model = whisper.load_model(self.model)

        # Load the microphone
        self.microphone = self.microphone_class()


def transcribeRT(task: TranscribeRTTask) -> TranscribeResult:
    task.load_resources()

    result = TranscribeResult()

    # Run for 1 minute
    for window_number in range(12):
        log.debug("Fetching Samples")
        samples = task.microphone.get()

        # Transcribe samples
        task = TranscribeTask(
            samples=samples,
            batch_name=f"Transcribe_RT::{timestamp()}",
            sequence_num=window_number,
            model=task.model,
        )
        log.info("Transcribing a %s second block", WINDOW_SECONDS)
        result.text += " " + transcribe(task)["text"]

        # Log timing
        log.debug("Transcribed in %s ms", task.duration)

        # Dump the task to disk
        task.dump()

    task.complete(result)
    return result

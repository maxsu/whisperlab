from pathlib import Path

from pytest import fixture

from whisperlab.tasks import Task
from whisperlab.transcribe import (
    transcribe,
    DEFAULT_TRANSCRIPTION_MODEL,
    TranscribeTask,
)


# Fixtures --------------------------------------------------------------------


@fixture
def poem_file() -> Path:
    return Path("tests/data/poem_sappho_58_by_Jameson_Fitzpatrick.mp3")


@fixture
def poem(poem_file) -> TranscribeTask:
    return TranscribeTask(audio_file=poem_file)


@fixture
def empty_file() -> TranscribeTask:
    return TranscribeTask(audio_file=Path("tests/data/empty_file.mp3"))


# Test Task Creation ----------------------------------------------------------


def test_create_simple_request(poem: Task, poem_file: Path):
    assert poem.args == {}
    assert poem.model == DEFAULT_TRANSCRIPTION_MODEL
    assert poem.audio_file == poem_file


# Test Transcribe Task --------------------------------------------------------


def test_transcribe_empty_file_to_empty_text(empty_file: Task):
    result = transcribe(empty_file)
    assert result.text == ""


def test_transcribe_poem(poem):
    result = transcribe(poem)
    assert result.text

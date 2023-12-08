from pathlib import Path

from pytest import fixture

from whisperlab.run_whisper import WhisperRequest, DEFAULT_WHISPER_MODEL
from whisperlab.run_whisper import run_whisper


# Fixtures --------------------------------------------------------------------


@fixture
def audio_file():
    return Path("tests/data/poem_sappho_58_by_Jameson_Fitzpatrick.mp3")


@fixture
def simple_request(audio_file):
    return WhisperRequest(audio_file=audio_file)


@fixture
def empty_file_request():
    return WhisperRequest(audio_file=Path("tests/data/empty_file.mp3"))


# Test Request ---------------------------------------------------------------


def test_create_simple_request(simple_request, audio_file):
    assert simple_request.args == {}
    assert simple_request.model == DEFAULT_WHISPER_MODEL
    assert simple_request.audio_file == audio_file


# Test Transcribe ------------------------------------------------------------


def test_empty_file_transcribes_to_empty_text(empty_file_request):
    # Expect an exception to be raised
    result = run_whisper(empty_file_request)
    assert result["text"] == ""


def test_run_whisper(simple_request):
    result = run_whisper(simple_request)
    assert result["text"]

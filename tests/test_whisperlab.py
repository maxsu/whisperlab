import logging as log

import whisperlab
from whisperlab.logging import config_log

config_log()


def test_whisper():
    assert whisperlab is not None


def test_transcribe():
    request = whisperlab.WhisperRequest(
        audio_file="tests/data/poem_sappho_58_by_Jameson_Fitzpatrick.mp3",
    )

    result = whisperlab.run_whisper(request)

    assert result["text"] is not None

    log.info(result["text"])

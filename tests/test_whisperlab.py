import logging as log

import whisperlab
from whisperlab.logging import config_log

config_log()


def test_whisper():
    assert whisperlab is not None

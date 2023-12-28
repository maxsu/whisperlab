import atexit

import datetime
import sounddevice
import numpy as np

from whisperlab.transcribe import transcribe, TranscribeTask
from whisperlab.logging import config_log
from whisperlab.time import time_ms, timestamp
from whisperlab.audio import SAMPLES_PER_SECOND


log = config_log(debug=True)

WINDOW_SECONDS = 5
SAMPLES_PER_WINDOW = SAMPLES_PER_SECOND * WINDOW_SECONDS


def Usecase():
    # Setup the microphone
    stream = sounddevice.InputStream(
        channels=1,
        blocksize=SAMPLES_PER_WINDOW,
        samplerate=SAMPLES_PER_SECOND,
    )
    stream.start()

    trancription = ""

    # log transcription at exit (ctrl-c)
    def exit_handler():
        log.info("Transcription: %s", trancription)

    atexit.register(exit_handler)

    # Run for 1 minute
    for frame in range(12):
        time = time_ms()
        log.debug("Fetching Samples")
        samples, overflowed = stream.read(SAMPLES_PER_WINDOW)
        if overflowed:
            raise Exception("Unexpected overflow")
        samples = samples[:, 0]  # Unwrap channel 1
        log.debug(
            "Fetched %s samples in %s ms",
            len(samples),
            time_ms() - time,
        )

        log.info("Transcribing a %s second block", WINDOW_SECONDS)

        task = TranscribeTask(
            batch=f"Transcribe_RT::{timestamp()}",
            sequence=frame,
            samples=samples,
        )
        result = transcribe(task)

        log.debug("Transcribed in %s ms", time_ms() - task.creation_time)

        trancription += " " + result["text"]

    log.info("Transcription: %s", trancription)


if __name__ == "__main__":
    Usecase()

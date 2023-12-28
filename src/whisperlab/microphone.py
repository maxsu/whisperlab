"""
Module: whisperlab.microphone

Plot the live microphone signal(s) with matplotlib.

This module provides functionality to visualize live microphone input using 
matplotlib and numpy. It includes utilities for audio processing, 
frame monitoring, and graphical display.
"""

import atexit
import logging

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice

import whisperlab.logging
from whisperlab.time import time_ms

log = whisperlab.logging.config_log(debug=True)


# Constants ===================================================================

# Unit conversion constants
MS_PER_SECOND = 1_000
SECONDS_PER_MS = 0.001

# Audio constants
CHANNEL = 1  # Input channel to use (assume 1 microphone channel)
SAMPLES_PER_SECOND = 44_100  # Sample rate (in samples/sec)
SAMPLES_PER_MS = 44.1  # Samples per millisecond (in samples/ms) = SAMPLES_PER_SECOND * SECONDS_PER_MS

# Plot constants
WINDOW_SECONDS = 8  # Width of plot window (in seconds)
FRAMES_PER_SECOND = 20  # Frame rate of the plot display (in Hz)
DOWNSAMPLE = 10  # Display every Nth sample
SAMPLES_PER_WINDOW = 35_280
# = int(SAMPLES_PER_SECOND * WINDOW_SECONDS / DOWNSAMPLE)
MS_PER_FRAME = 50  # Frame rate of the plot display (in ms) = 1000 / FRAMES_PER_SECOND
PLOT_SAMPLES_PER_MS = 4.41  # = SAMPLES_PER_MS / DOWNSAMPLE
RAW_SAMPLES_PER_FRAME = 2205  # = int(SAMPLES_PER_MS * MS_PER_FRAME)


# Model =======================================================================


def roll(buffer: np.ndarray, samples: np.ndarray):
    """Roll new samples into a numpy array. Use downsampling."""

    if len(samples) > len(buffer):
        return samples[-len(buffer) :]
    left_shift = -len(samples)
    buffer = np.roll(buffer, left_shift)
    buffer[left_shift:] = samples
    return buffer


class PlotBuffer:
    """A model of the plot signal buffer."""

    buffer = np.zeros(SAMPLES_PER_WINDOW)

    def get(self):
        return self.buffer

    def put(self, audio_samples):
        """Convert the audio samples to a plot buffer."""
        plot_samples = audio_samples[::DOWNSAMPLE, CHANNEL - 1]
        self.buffer = roll(self.buffer, plot_samples)


# View ========================================================================


frame_sample_counter = 0
plot_timer = time_ms()


def frame_monitor(update_func):
    """Monitor frame interval and samples."""

    def wrapped_update_func(self, frame):
        if log.level == logging.DEBUG:
            global frame_sample_counter
            global plot_timer
            t1 = time_ms()
            interval_ms = t1 - plot_timer
            plot_timer = t1
            expected_samples = round(interval_ms * PLOT_SAMPLES_PER_MS)
            discrepancy = frame_sample_counter - expected_samples
            log.debug(
                "Frame %s: %s ms, %s samples. "
                "Expected: %s samples, discrepancy: %s samples.",
                frame,
                interval_ms,
                frame_sample_counter,
                expected_samples,
                discrepancy,
            )
            frame_sample_counter = 0
        return update_func(self, frame)

    return wrapped_update_func


class View:
    """
    Plot the live microphone signal with matplotlib.
    """

    lines: list[plt.Line2D]
    animation: FuncAnimation
    model: PlotBuffer

    def __init__(self, model):
        self.model = model

    @frame_monitor
    def update(self, frame):
        """Update the plot each frame."""
        self.lines[0].set_ydata(self.model.get())
        return self.lines

    def start(self):
        """Start the animation"""
        figure, axis = plt.subplots()
        self.lines = axis.plot(self.model.get())
        axis.set(xlim=(0, SAMPLES_PER_WINDOW), ylim=(-1, 1))
        figure.tight_layout(pad=0)  # Scale plot to fit the window
        self.animation = FuncAnimation(
            figure,
            self.update,
            interval=1000 // FRAMES_PER_SECOND,
            blit=True,
        )
        plt.show()

    def stop(self):
        """Stop the animation"""
        plt.close()


# Recorders ===================================================================


def callback_monitor(callback):
    """Monitor callback execution time and samples."""

    def wrapped_callback(self, indata, frames, time, status):
        if log.level == logging.DEBUG:
            global frame_sample_counter
            frame_sample_counter += len(indata)
            start_time = time_ms()
            log.debug("Starting update")
            log.debug("<-- Roll %s samples", len(indata))
            callback(self, indata, frames, time, status)
            log.debug("Update Completed in %s ms", time_ms() - start_time)
        else:
            callback(self, indata, frames, time, status)

    return wrapped_callback


class Recorder:
    """Listen to the microphone and record samples to a data model."""

    stream: sounddevice.InputStream
    model: PlotBuffer
    blocksize: int | None

    def __init__(self, model, blocksize=None):
        self.stream = sounddevice.InputStream(
            callback=self.callback,
            blocksize=blocksize,
        )
        self.model = model
        self.blocksize = blocksize

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

    @callback_monitor
    def callback(self, samples, frames, time, status):
        self.model.put(samples)


def FrameBlockRecorder(model):
    return Recorder(model, blocksize=RAW_SAMPLES_PER_FRAME)


def FiveSecondBlockRecorder(model):
    return Recorder(model, blocksize=SAMPLES_PER_SECOND * 5)


# Controller ==================================================================


DEFAULT_STREAM = Recorder


class App:
    """A controller for the application."""

    def __init__(self, view_class=View, stream_class=Recorder):
        self.model = PlotBuffer()
        self.view = view_class(model=self.model)
        self.stream = stream_class(model=self.model)

    def start(self):
        """Start the application resources."""
        # Stop the application cleanly on exit
        atexit.register(self.stop)
        log.info("Starting stream")
        self.stream.start()
        log.info("Starting view")
        self.view.start()
        log.info("Application started")

    def stop(self):
        """Stop the application resources."""
        log.info("Stopping stream")
        self.stream.stop()
        log.info("Stopping view")
        self.view.stop()
        log.info("Application stopped")


if __name__ == "__main__":
    app = App(stream_class=FrameBlockRecorder)
    app.start()

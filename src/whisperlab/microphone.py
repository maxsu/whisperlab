#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

"""
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


CHANNEL = 1  # Input channel to use (assume 1 microphone channel)
DEVICE = None  # Input device (use default device)
SAMPLERATE = sd.query_devices(kind="input")["default_samplerate"]
# Sample rate (in samples/sec) (use default)
BLOCKSIZE = None  # Block size (in samples) (use default)
DOWNSAMPLE = 10  # Display every Nth sample
INTERVAL = 30  # Minimum time between plot updates (in ms)
WINDOW = 200  # Width of plot window (in ms)
LENGTH = int(WINDOW * SAMPLERATE / (1000 * DOWNSAMPLE))  # Samples in signal window
MAPPING = [0]

q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::DOWNSAMPLE, MAPPING])


def get_plot(length):
    """Create a new plot with required buffer length."""
    plotdata = np.zeros((length, 1))
    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    fig.tight_layout(pad=0)

    return fig, lines, plotdata


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines


if __name__ == "__main__":
    # Put lines and plotdata in scope for the update_plot function
    fig, lines, plotdata = get_plot(LENGTH)

    stream = sd.InputStream(
        device=DEVICE,
        channels=CHANNEL,
        samplerate=SAMPLERATE,
        callback=audio_callback,
    )
    ani = FuncAnimation(fig, update_plot, interval=INTERVAL, blit=True)
    with stream:
        plt.show()
    while True:
        pass

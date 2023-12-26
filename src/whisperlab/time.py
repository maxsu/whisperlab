import time


def time_ms():
    """
    Get the current time in milliseconds.

    Returns:
        int: The current time in milliseconds since epoch.
    """
    return int(round(time.time() * 1000))

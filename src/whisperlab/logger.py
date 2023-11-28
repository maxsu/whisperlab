"""
Logging Module

This module provides a simple logging interface that can be used to log to the console and/or a file.

Example Usage:

1. Setup in Main.py

    from Service import logger
    logger.config(verbose=True, structured=True, log_file="log.txt")

2. Use in Module.py

    from Service import logger
    logger.info("This is an info message")
"""

from contextlib import contextmanager
import json
from logging import FileHandler, Formatter, StreamHandler,  DEBUG, INFO, Logger
import sys

from pathlib import Path


# Global Logger
logger = None
info, debug, warning, error = None, None, None, None


def config(verbose:bool=False, structured:bool=False, log_file:str=None):
    global logger, info, debug, warning, error

    # Create the root logger
    logger = Logger("main")
    info, debug, warning, error = logger.info, logger.debug, logger.warning, logger.error

    # Set the logger level
    if verbose:
        logger.setLevel(DEBUG)
    else:
        logger.setLevel(INFO)
   
    # Build the console handler
    if structured:
        formatter = JsonFormatter()
    elif verbose:
        formatter = DefaultFormatter(LONG_FORMAT, style="{")   
    else:
        formatter = DefaultFormatter(SHORT_FORMAT, style="{")   
    console_sink = StreamHandler(sys.stdout)
    console_sink.setFormatter(formatter)
    logger.addHandler(console_sink)

    # Fork the log to a file
    if log_file:
        # Build the file handler
        file_sink = FileHandler(log_file)
        file_sink.setFormatter(formatter)
        logger.addHandler(file_sink)


# Formats
SHORT_FORMAT = "{asctime} [{levelname}] {message}"

LONG_FORMAT = """{asctime} [{levelname}] {name}:{filename}.{funcName}:L{lineno}. {message}
===================="""

JSON_FORMAT = [
    "name",
    "msg",
    "module",
    "funcName",
    "lineno",
    "created",
    "args",
    "levelno",
]
# See: https://docs.python.org/3/library/logging.html#logrecord-attributes


# Formatters
class DefaultFormatter(Formatter):
    def format(self, record):
        # Shorten the level name
        record.levelname = record.levelname[0]
        return super(DefaultFormatter, self).format(record)

class JsonFormatter(Formatter):
    def format(self, record):
        # Filter the log record to only include the keys we want
        filtered_record = {k: v for k, v in record.__dict__.items() if k in JSON_FORMAT}
        # Convert the log record to a JSON string
        json_record = json.dumps(filtered_record)
        return json_record

    def parse(self, log_file: Path)->list[dict]:
        # Read the log file into a list of JSON dictionaries
        results = [json.loads(L) for L in log_file.read_text().split("\n") if L.strip()]
        return results
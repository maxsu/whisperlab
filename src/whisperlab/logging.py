"""
Logging Config

This module pulls the logging config from the tool.logging section
of the project's pyproject.toml file.
"""

import logging
import logging518.config
from pathlib import Path

CONFIG_FILE = "pyproject.toml"
LOG_CONFIGURED = False


def config_log(debug=False):
    """
    Configure the logging module and provide the main logger.

    This pulls the logging config from the tool.logging section of the
    project's pyproject.toml file.

    It is safe to call this function multiple times. It will only configure
    logging on the first call.

    Args:
        debug (bool): Whether to set the log level to debug.

    Returns:
        logging.Logger: The main logger.
    """

    global LOG_CONFIGURED

    # Get the main logger
    log = logging.getLogger("main")

    # Only configure logging once
    if not LOG_CONFIGURED:
        LOG_CONFIGURED = True

        # Load the logging config from the project's pyproject.toml file
        logging518.config.fileConfig(CONFIG_FILE)

        # Set the log level to debug if requested
        if debug:
            log.setLevel(logging.DEBUG)

    return log


class Formatter(logging.Formatter):
    """
    The project's default formatter

    Attributes ----------------------------------------------------------------

    This formatter adds the following attribute to log records:

    relpath: The relative path of file relative to the project root directory.
        This facilitates log file analysis and lets IDE users ctrl+click on the
        path to navigate to the line.

    For other available Attributes, see:
        https://docs.python.org/3/library/logging.html#logrecord-attributes

    """

    def format(self, record):
        record.relpath = Path(record.pathname).relative_to(Path.cwd())
        record.shortlvl = record.levelname[0]
        return super().format(record)

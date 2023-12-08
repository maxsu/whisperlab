"""
Logging Config

This module pulls the logging config from the tool.logging section
of the project's pyproject.toml file.
"""

import logging
import logging518.config
from pathlib import Path

CONFIG_FILE = "pyproject.toml"


def config_log():
    # Load the logging config from the project's pyproject.toml file
    logging518.config.fileConfig(CONFIG_FILE)


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

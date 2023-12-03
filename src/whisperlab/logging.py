"""
Logging Config

This module pulls the logging config from the tool.logging section
of the project's pyproject.toml file.
"""

import logging
import logging.config
from pathlib import Path
import toml

CONFIG_FILE = "pyproject.toml"


def config_log():
    # Parse logging config dictionary
    config = toml.load(CONFIG_FILE)['tool']['logging']
    # Load logging config
    logging.config.dictConfig(config)

    return logging.getLogger('main')


class DefaultFormatter(logging.Formatter):
    """
    A custom formatter that adds `relpath` and `shortlvl` attribute
    to log records.

    See: https://docs.python.org/3/library/logging.html#logrecord-attributes
    """

    def format(self, record):
        record.relpath = Path(record.pathname).relative_to(Path.cwd())
        record.shortlvl = record.levelname[0]
        return super().format(record)


class JsonFormatter(DefaultFormatter):
    """
    A custom formatter that formats logging records as json strings.

    See: https://docs.python.org/3/library/logging.html#logrecord-attributes
    """

    include_keys = [
        # Useful Fields
        "module",
        "funcName",
        "lineno",
        "relpath",
        "shortlvl",
        # Required Fields
        # (Excluding these causes an exception in the logging module)
        "name",
        "args",
        "created",
        "levelno",
        "msg",
    ]

    def format(self, record):
        # Filter the log record
        record = {k: record.__dict__[k] for k in self.include_keys}
        # Dump the record to json
        return json.dumps(record)

    def parse(self, log_file: Path) -> list[dict]:
        # Read the log lines
        lines = log_file.read_text().split("\n")
        # Drop empty lines
        lines = [x for x in lines if x.strip()]
        # Parse lines to json dictionaries
        return [json.loads(x) for x in lines]


# Manual Test
if __name__ == "__main__":
    config_log()
    log = logging.getLogger("main")
    log.info("This is an info message")
    log.debug("This is a debug message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.info("This is an exception message:")
    log.exception(ValueError("ValueError: This is an exception message"))

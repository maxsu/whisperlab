import logging


def test_live_logging():
    # Contract: The logger is visible in the console
    log = logging.getLogger()
    log.info("This is an info message")
    log.debug("This is a debug message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.info("This is an exception message")
    log.exception(ValueError("This is an exception message"))

from whisperlab import logger

import logging as log

logger.config_log()


def test_logger():
    log.info("This is an info message")
    log.debug("This is a debug message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.info("This is an exception message")
    log.exception(ValueError("This is an exception message"))

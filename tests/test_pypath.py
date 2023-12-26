import sys
import os

import logging
import whisperlab.logging

log = whisperlab.logging.config_log()

log.info("Running tests/test_main.py")


def test_diagnose_environment():
    # Check which Python interpreter is being used
    python_executable = sys.executable
    log.info(f"Python Executable: {python_executable}")

    # List the contents of the PATH variable
    path_variable = os.environ.get("PATH", "")
    log.info("PATH Variable Contents:")
    for path in path_variable.split(os.pathsep):
        log.info(f"  {path}")

    # Optionally, check if whisperlab

from pathlib import Path


LOG_FILES = Path().glob("**/*.log")


def clean_whisperlab():
    # Clean up log files
    for log_file in LOG_FILES:
        log_file.unlink()


if __name__ == "__main__":
    clean_whisperlab()

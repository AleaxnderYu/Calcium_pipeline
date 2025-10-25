"""
Logging configuration for the calcium imaging pipeline.
"""

import logging
import sys
from pathlib import Path


class TeeStream:
    """Stream that writes to both the original stream and a log file."""
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, data):
        self.original_stream.write(data)
        if self.log_file and not self.log_file.closed:
            self.log_file.write(data)
            self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()

    def isatty(self):
        """Return whether the original stream is a TTY."""
        return self.original_stream.isatty()

    def fileno(self):
        """Return file descriptor of original stream."""
        return self.original_stream.fileno()

    def __getattr__(self, name):
        """Delegate any other attributes to the original stream."""
        return getattr(self.original_stream, name)


def setup_logging(log_level: str = "INFO", log_file: Path = None, verbose: bool = False):
    """
    Configure logging for the entire application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        verbose: If True, set level to DEBUG
    """
    if verbose:
        log_level = "DEBUG"

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified) - mode='w' to overwrite on each start
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite instead of append
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Also redirect stdout and stderr to the log file
        # This captures any print() statements or other stdout/stderr output
        log_file_obj = open(log_file, 'a')  # Open in append mode since we already created it
        sys.stdout = TeeStream(sys.__stdout__, log_file_obj)
        sys.stderr = TeeStream(sys.__stderr__, log_file_obj)

    return root_logger

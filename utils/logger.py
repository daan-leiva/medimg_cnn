# utils/logger.py

import logging
from pathlib import Path
import sys

class Logger:
    """
    A lightweight logging utility that writes logs to both the console and a log file.
    Automatically creates a logs directory and prevents duplicate handler registration.
    """

    def __init__(self, log_dir="logs", log_file="app.log", level=logging.INFO):
        """
        Initializes the logger by setting up file and console handlers.

        Args:
            log_dir (str): Directory to store the log file.
            log_file (str): Name of the log file.
            level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        # Create logs directory if it doesn't exist
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create a logger instance with the log file name as identifier
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent messages from being passed to the root logger

        # Configure console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # Configure file output
        file_handler = logging.FileHandler(self.log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)

        # Add handlers only if they haven't been added yet
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    # Convenience methods to log at different severity levels
    def info(self, msg):
        """Log an informational message."""
        self.logger.info(msg)

    def warning(self, msg):
        """Log a warning message."""
        self.logger.warning(msg)

    def error(self, msg):
        """Log an error message."""
        self.logger.error(msg)

    def debug(self, msg):
        """Log a debug message."""
        self.logger.debug(msg)

    def critical(self, msg):
        """Log a critical message."""
        self.logger.critical(msg)
# utils/logger.py

import logging
from pathlib import Path
import sys

class Logger:
    def __init__(self, log_dir="logs", log_file="app.log", level=logging.INFO):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Avoid duplicate logs if root logger is configured

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # File handler
        file_handler = logging.FileHandler(self.log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)

        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def debug(self, msg): self.logger.debug(msg)
    def critical(self, msg): self.logger.critical(msg)

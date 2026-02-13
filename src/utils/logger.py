"""Logging configuration for WhatsApp-GPT application.

Uses Python's built-in logging with automatic filename/function name
in log output via format specifiers (no frame inspection needed).
"""

import logging
import os

from config import settings


class Logger:
    """Application logger with console and file output.
    
    Uses %(filename)s and %(funcName)s format specifiers for automatic
    caller identification — no frame inspection overhead.
    """

    def __init__(self, name: str = "WAHALogger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(settings.log_level)
        self.logger.propagate = False

        # Avoid duplicate handlers on re-import
        if not self.logger.handlers:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | file: %(filename)s | func: %(funcName)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            # Console (stdout)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            # File (log to logs/app.log)
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                f"{log_dir}/app.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        self.logger.debug(message, stacklevel=2)

    def info(self, message: str) -> None:
        self.logger.info(message, stacklevel=2)

    def warning(self, message: str) -> None:
        self.logger.warning(message, stacklevel=2)

    def error(self, message: str) -> None:
        self.logger.error(message, stacklevel=2)

    def critical(self, message: str) -> None:
        self.logger.critical(message, stacklevel=2)


logger = Logger()

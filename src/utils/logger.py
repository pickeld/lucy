import logging
import inspect
import os
from config import config


class Logger:
    def __init__(self, name: str = "WAHALogger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(config.log_level)
        self.logger.propagate = False

        # Avoid duplicate handlers
        if not self.logger.handlers:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            # Console (stdout)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            # File (log to logs/app.log or wherever you like)
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                f"{log_dir}/app.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _log(self, level: str, message: str) -> None:
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back
        if frame is not None:
            frame = frame.f_back
        if frame is not None:
            caller = frame.f_code.co_name
            filename = frame.f_code.co_filename.split("/")[-1]
            formatted = f"file: {filename} | func: {caller} | {message}"
        else:
            formatted = message
        getattr(self.logger, level)(formatted)

    def debug(self, message: str):
        self._log("debug", message)

    def info(self, message: str):
        self._log("info", message)

    def warning(self, message: str):
        self._log("warning", message)

    def error(self, message: str):
        self._log("error", message)

    def critical(self, message: str):
        self._log("critical", message)


logger = Logger()

import logging
import inspect
from config import config

class Logger:
    def __init__(self, name: str = "WAHALogger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(config.log_level)
        self.logger.propagate = False

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _log(self, level: str, message: str):
        # Get caller function name and filename (2 frames up)
        frame = inspect.currentframe().f_back.f_back
        caller = frame.f_code.co_name
        filename = frame.f_code.co_filename.split("/")[-1]
        formatted = f"file: {filename} | func: {caller} | {message}"
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
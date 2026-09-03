"""Consistent logging setup for CLI and library use."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


class _Formatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self, *, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.color:
            prefix = self.COLORS.get(record.levelname, "")
            if prefix:
                text = f"{prefix}{text}{self.RESET}"
        return text


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Attach a stderr handler (and optionally a file handler) to the root logger."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not _CONFIGURED:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(_Formatter(color=sys.stderr.isatty()))
        root.addHandler(stream)
        # yfinance/urllib3 are noisy at DEBUG and add nothing here.
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("yfinance").setLevel(logging.WARNING)
        logging.getLogger("peewee").setLevel(logging.WARNING)
        _CONFIGURED = True

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s")
        )
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

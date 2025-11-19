"""Logging helpers that honor application configuration."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Optional

from src.core.config import AppConfig, ConfigManager

_CONFIGURED = False
_LOCK = Lock()


def configure_logging(config: Optional[AppConfig] = None) -> None:
    """Configure the root logger using settings.yaml values."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    with _LOCK:
        if _CONFIGURED:
            return
        settings = config or ConfigManager.load().settings
        log_settings = settings.logging
        log_settings.ensure_directories()

        formatter = logging.Formatter(log_settings.format)
        file_handler = logging.FileHandler(log_settings.file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        handlers = [file_handler]
        if log_settings.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        for handler in handlers:
            root_logger.addHandler(handler)

        level = getattr(logging, log_settings.level.upper(), logging.INFO)
        root_logger.setLevel(level)
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger configured for this application."""
    configure_logging()
    return logging.getLogger(name)


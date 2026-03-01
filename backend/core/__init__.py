"""
Core module initialization.

This module provides core configuration, logging, and utility functions
for the AI Research & Code Copilot application.
"""

from backend.core.config import Settings, settings, get_settings
from backend.core.logging import (
    setup_logging,
    get_logger,
    LoggerMixin,
    log_execution,
    LogLevel
)

__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "setup_logging",
    "get_logger",
    "LoggerMixin",
    "log_execution",
    "LogLevel"
]

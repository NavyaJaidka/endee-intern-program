"""
Logging configuration module for the AI Research & Code Copilot.

Provides centralized logging setup with file and console handlers,
support for structured logging, and custom log levels.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        rotation: Log file rotation size
        retention: Log file retention period
        format_string: Custom format string for logs
    """
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add file handler with rotation
    logger.add(
        log_file,
        level=log_level,
        format=format_string,
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logging initialized at {log_level} level")
    logger.info(f"Log file: {log_file}")


def get_logger(name: str = __name__):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


# Pre-configured loggers for different modules
class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                self.logger = self.get_logger()
    """
    
    @property
    def logger(self):
        """Get logger bound to the class name."""
        return get_logger(self.__class__.__name__)


# Decorator for logging function execution
def log_execution(logger=None):
    """
    Decorator to log function execution.
    
    Usage:
        @log_execution()
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            log.debug(f"Executing {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                log.debug(f"Completed {func.__name__}")
                return result
            except Exception as e:
                log.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator


# Context manager for temporary log level
class LogLevel:
    """
    Context manager to temporarily change log level.
    
    Usage:
        with LogLevel("DEBUG"):
            # Debug logging here
            pass
    """
    
    def __init__(self, level: str):
        self.level = level
        self.previous_level = None
    
    def __enter__(self):
        self.previous_level = logger.level
        logger.level = self.level
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.level = self.previous_level

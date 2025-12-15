#!/usr/bin/env python3
"""
Comprehensive logging configuration and utilities for piedomains.

This module provides centralized logging configuration with proper formatters,
handlers, and log levels for the entire piedomains package. It supports both
console and file logging with configurable log levels and formats.

Example:
    Basic usage:
        >>> from piedomains.piedomains_logging import get_logger
        >>> logger = get_logger()
        >>> logger.info("Processing domain classification")

    With custom configuration:
        >>> from piedomains.piedomains_logging import configure_logging
        >>> configure_logging(level="DEBUG", console_format="detailed")
        >>> logger = get_logger()
        >>> logger.debug("Detailed debug information")
"""

import logging
import sys
from pathlib import Path

# Default log formats
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
SIMPLE_FORMAT = "%(levelname)s - %(message)s"

# Global logger configuration state
_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance for piedomains with proper configuration.

    Args:
        name (str, optional): Logger name. If None, uses 'piedomains' as the base logger.
                             For module-specific loggers, pass __name__.

    Returns:
        logging.Logger: Configured logger instance with appropriate handlers and formatting.

    Example:
        >>> # Get the main piedomains logger
        >>> logger = get_logger()
        >>> logger.info("Main application log")

        >>> # Get a module-specific logger
        >>> logger = get_logger(__name__)
        >>> logger.debug("Module-specific debug info")
    """
    if not _configured:
        configure_logging()

    logger_name = name or "piedomains"
    return logging.getLogger(logger_name)


def configure_logging(
    level: str | int = "INFO",
    console_format: str = "default",
    file_path: str | None = None,
    file_level: str | int = "DEBUG",
    force_reconfigure: bool = False,
) -> None:
    """
    Configure logging for the piedomains package with comprehensive options.

    Args:
        level (Union[str, int]): Console logging level. Can be string ('DEBUG', 'INFO', etc.)
                                or logging constant (logging.INFO, etc.). Defaults to 'INFO'.
        console_format (str): Console log format style. Options:
                             - 'default': Standard format with timestamp and level
                             - 'detailed': Detailed format with module/function/line info
                             - 'simple': Simple format with just level and message
                             Defaults to 'default'.
        file_path (str, optional): Path to log file. If provided, enables file logging.
                                  Directory will be created if it doesn't exist.
        file_level (Union[str, int]): File logging level (if file_path provided).
                                     Defaults to 'DEBUG' for comprehensive file logs.
        force_reconfigure (bool): If True, reconfigure even if already configured.
                                 Defaults to False.

    Raises:
        ValueError: If an invalid format style or log level is provided.
        OSError: If the log file directory cannot be created.

    Example:
        >>> # Basic console logging
        >>> configure_logging(level="DEBUG")

        >>> # Console + file logging with detailed format
        >>> configure_logging(
        ...     level="INFO",
        ...     console_format="detailed",
        ...     file_path="/var/log/piedomains/app.log",
        ...     file_level="DEBUG"
        ... )
    """
    global _configured

    if _configured and not force_reconfigure:
        return

    # Convert string levels to logging constants
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper(), logging.DEBUG)

    # Select console format
    format_map = {
        "default": DEFAULT_FORMAT,
        "detailed": DETAILED_FORMAT,
        "simple": SIMPLE_FORMAT,
    }

    if console_format not in format_map:
        raise ValueError(
            f"Invalid format style: {console_format}. "
            f"Must be one of: {list(format_map.keys())}"
        )

    console_formatter = logging.Formatter(format_map[console_format])

    # Get the root piedomains logger
    logger = logging.getLogger("piedomains")
    logger.setLevel(logging.DEBUG)  # Allow all levels, handlers filter

    # Clear existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if file_path:
        try:
            # Ensure log directory exists
            log_dir = Path(file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT))
            logger.addHandler(file_handler)

            logger.info(f"File logging enabled: {file_path}")

        except (OSError, PermissionError) as e:
            # Fallback to console-only logging if file fails
            logger.warning(f"Could not set up file logging at {file_path}: {e}")

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    _configured = True
    logger.debug(
        f"Logging configured: console_level={logging.getLevelName(level)}, "
        f"format={console_format}"
    )


def set_level(level: str | int) -> None:
    """
    Change the logging level for all existing piedomains loggers.

    Args:
        level (Union[str, int]): New logging level. Can be string ('DEBUG', 'INFO', etc.)
                                or logging constant (logging.INFO, etc.).

    Example:
        >>> set_level("DEBUG")  # Enable debug logging
        >>> set_level(logging.WARNING)  # Only warnings and errors
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("piedomains")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(level)
            break

    logger.debug(f"Console logging level changed to {logging.getLevelName(level)}")


def get_effective_level() -> str:
    """
    Get the current effective logging level for console output.

    Returns:
        str: Current logging level name (e.g., 'INFO', 'DEBUG', 'WARNING').

    Example:
        >>> current_level = get_effective_level()
        >>> print(f"Current log level: {current_level}")
        Current log level: INFO
    """
    logger = logging.getLogger("piedomains")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            return logging.getLevelName(handler.level)

    return "INFO"  # Default fallback


def disable_logging() -> None:
    """
    Disable all piedomains logging output.

    This is useful for testing or when running in quiet mode.
    Use configure_logging() to re-enable logging.

    Example:
        >>> disable_logging()  # Silence all piedomains logs
        >>> # ... run operations silently ...
        >>> configure_logging()  # Re-enable logging
    """
    logger = logging.getLogger("piedomains")
    logger.setLevel(logging.CRITICAL + 1)  # Level higher than any message


def is_debug_enabled() -> bool:
    """
    Check if DEBUG level logging is currently enabled.

    Returns:
        bool: True if DEBUG logging is enabled, False otherwise.

    Example:
        >>> if is_debug_enabled():
        ...     logger.debug("This will only run if debug is enabled")
    """
    logger = logging.getLogger("piedomains")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            return handler.level <= logging.DEBUG
    return False

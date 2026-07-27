#!/usr/bin/env python3
"""Comprehensive logging configuration and utilities for piedomains.

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

import json
import logging
import os
import sys
from pathlib import Path

# Default log formats
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
SIMPLE_FORMAT = "%(levelname)s - %(message)s"

# Global logger configuration state
_configured = False

# Fields LogRecord always carries; anything else came from `extra=` and is
# forwarded into the JSON payload.
_STANDARD_RECORD_FIELDS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}

#: Context merged into every record — set by `bind_context`.
_context: dict[str, object] = {}


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line.

    Any keyword passed via ``extra=`` — ``run_id``, ``domain``, ``stage``,
    ``error_code`` — is promoted to a top-level key so log lines can be
    filtered and correlated with the run report.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a record to a single-line JSON object.

        Args:
            record: The record to render.

        Returns:
            str: A JSON object, newline-free.
        """
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_context)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS:
                payload[key] = value
        if record.exc_info:
            payload["exc_type"] = getattr(record.exc_info[0], "__name__", None)
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ContextFilter(logging.Filter):
    """Attach the bound context to records so text formats can use it too."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Merge bound context onto the record.

        Args:
            record: The record being emitted.

        Returns:
            bool: Always True; the record is never dropped.
        """
        for key, value in _context.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def bind_context(**fields: object) -> None:
    """Bind fields onto every subsequent log record.

    Used to thread ``run_id`` through a batch so that log lines can be joined
    against the run report.

    Args:
        **fields: Key/value pairs to attach. A value of ``None`` unbinds.
    """
    for key, value in fields.items():
        if value is None:
            _context.pop(key, None)
        else:
            _context[key] = value


def clear_context() -> None:
    """Remove all bound context fields."""
    _context.clear()


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for piedomains with proper configuration.

    Args:
        name: Logger name. If None, uses 'piedomains' as the base logger.
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
    """Configure logging for the piedomains package with comprehensive options.

    Args:
        level: Console logging level. Can be string ('DEBUG', 'INFO', etc.)
                                or logging constant (logging.INFO, etc.). Defaults to 'INFO'.
        console_format: Console log format style. Options:
                             - 'default': Standard format with timestamp and level
                             - 'detailed': Detailed format with module/function/line info
                             - 'simple': Simple format with just level and message
                             Defaults to 'default'.
        file_path: Path to log file. If provided, enables file logging.
                                  Directory will be created if it doesn't exist.
        file_level: File logging level (if file_path provided).
                                     Defaults to 'DEBUG' for comprehensive file logs.
        force_reconfigure: If True, reconfigure even if already configured.
                                 Defaults to False.


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


    Raises:
        ValueError: If an argument is invalid.
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

    # PIEDOMAINS_LOG_FORMAT=json opts into machine-readable output; the default
    # stays human-readable so interactive use is unchanged.
    env_format = os.environ.get("PIEDOMAINS_LOG_FORMAT", "").strip().lower()
    if env_format and console_format == "default":
        console_format = env_format

    if console_format == "json":
        console_formatter: logging.Formatter = JsonFormatter()
    elif console_format in format_map:
        console_formatter = logging.Formatter(format_map[console_format])
    else:
        raise ValueError(
            f"Invalid format style: {console_format}. "
            f"Must be one of: {[*format_map, 'json']}"
        )

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
    console_handler.addFilter(_ContextFilter())
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
    """Change the logging level for all existing piedomains loggers.

    Args:
        level: New logging level. Can be string ('DEBUG', 'INFO', etc.)
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
    """Get the current effective logging level for console output.

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
    """Disable all piedomains logging output.

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
    """Check if DEBUG level logging is currently enabled.

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

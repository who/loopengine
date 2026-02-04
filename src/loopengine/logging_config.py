"""Structured logging configuration for LoopEngine.

Configurable via environment variables:
- LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR). Default: INFO
- LOG_FORMAT: Set format ('text' or 'json'). Default: text

Usage:
    from loopengine.logging_config import configure_logging
    configure_logging()  # Call once at application startup
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import UTC, datetime
from typing import Any, ClassVar


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Produces JSON lines with consistent field ordering for easy parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON string with log data.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add source location for debug and error
        if record.levelno in (logging.DEBUG, logging.ERROR, logging.CRITICAL):
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if any
        extra_keys = set(record.__dict__.keys()) - {
            "name",
            "msg",
            "args",
            "created",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "msecs",
            "relativeCreated",
            "taskName",
        }
        if extra_keys:
            log_data["extra"] = {k: record.__dict__[k] for k in extra_keys}

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Standard text formatter with configurable detail level.

    Format: TIMESTAMP LEVEL [LOGGER] MESSAGE
    For DEBUG/ERROR: includes file:line in source
    """

    LEVEL_COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold red
    }
    RESET: ClassVar[str] = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        """Initialize formatter.

        Args:
            use_colors: Whether to use ANSI colors in output.
        """
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text.

        Args:
            record: Log record to format.

        Returns:
            Formatted string.
        """
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname

        if self.use_colors:
            color = self.LEVEL_COLORS.get(level, "")
            level_str = f"{color}{level:8s}{self.RESET}"
        else:
            level_str = f"{level:8s}"

        # Shorten logger name for readability
        logger_name = record.name
        if logger_name.startswith("loopengine."):
            logger_name = logger_name[11:]  # Remove 'loopengine.' prefix

        message = record.getMessage()

        # Base format
        parts = [f"{timestamp} {level_str} [{logger_name}] {message}"]

        # Add source location for debug and error
        if record.levelno in (logging.DEBUG, logging.ERROR, logging.CRITICAL):
            parts.append(f" ({record.filename}:{record.lineno})")

        # Add exception if present
        if record.exc_info:
            parts.append(f"\n{self.formatException(record.exc_info)}")

        return "".join(parts)


def get_log_level() -> int:
    """Get log level from environment.

    Reads LOG_LEVEL environment variable. Valid values:
    DEBUG, INFO, WARNING, ERROR, CRITICAL

    Returns:
        Logging level constant (e.g., logging.INFO).
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_name, logging.INFO)


def get_log_format() -> str:
    """Get log format from environment.

    Reads LOG_FORMAT environment variable. Valid values: text, json

    Returns:
        Format string ('text' or 'json').
    """
    format_name = os.environ.get("LOG_FORMAT", "text").lower()
    if format_name not in ("text", "json"):
        return "text"
    return format_name


def configure_logging(
    level: int | None = None,
    format_type: str | None = None,
    use_colors: bool = True,
) -> None:
    """Configure logging for the application.

    Should be called once at application startup before any logging.

    Args:
        level: Log level (use logging.DEBUG, logging.INFO, etc.)
               If None, reads from LOG_LEVEL env var.
        format_type: Output format ('text' or 'json').
                     If None, reads from LOG_FORMAT env var.
        use_colors: Whether to use colors in text format (only if stderr is TTY).
    """
    if level is None:
        level = get_log_level()
    if format_type is None:
        format_type = get_log_format()

    # Create handler
    handler = logging.StreamHandler(sys.stderr)

    # Set formatter based on format type
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter(use_colors=use_colors)

    handler.setFormatter(formatter)

    # Configure root logger for loopengine namespace
    root_logger = logging.getLogger("loopengine")
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate logs)
    root_logger.propagate = False

    # Also configure uvicorn access log if present (for API requests)
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers.clear()
    uvicorn_access.addHandler(handler)
    uvicorn_access.setLevel(level)
    uvicorn_access.propagate = False

    # Log configuration at debug level
    root_logger.debug(
        "Logging configured: level=%s, format=%s",
        logging.getLevelName(level),
        format_type,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module.

    Convenience function that ensures consistent naming.

    Args:
        name: Module name (typically __name__).

    Returns:
        Logger instance.
    """
    # Ensure logger is under loopengine namespace if not already
    if not name.startswith("loopengine"):
        name = f"loopengine.{name}"
    return logging.getLogger(name)

"""Tests for logging configuration module."""

from __future__ import annotations

import json
import logging
import os
from io import StringIO
from unittest.mock import patch

from loopengine.logging_config import (
    JSONFormatter,
    TextFormatter,
    configure_logging,
    get_log_format,
    get_log_level,
    get_logger,
)


class TestGetLogLevel:
    """Tests for get_log_level function."""

    def test_default_is_info(self) -> None:
        """Default log level should be INFO."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove LOG_LEVEL if it exists
            os.environ.pop("LOG_LEVEL", None)
            assert get_log_level() == logging.INFO

    def test_debug_level(self) -> None:
        """LOG_LEVEL=DEBUG should return logging.DEBUG."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            assert get_log_level() == logging.DEBUG

    def test_warning_level(self) -> None:
        """LOG_LEVEL=WARNING should return logging.WARNING."""
        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            assert get_log_level() == logging.WARNING

    def test_warn_alias(self) -> None:
        """LOG_LEVEL=WARN should work as alias for WARNING."""
        with patch.dict(os.environ, {"LOG_LEVEL": "WARN"}):
            assert get_log_level() == logging.WARNING

    def test_error_level(self) -> None:
        """LOG_LEVEL=ERROR should return logging.ERROR."""
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            assert get_log_level() == logging.ERROR

    def test_case_insensitive(self) -> None:
        """Log level should be case insensitive."""
        with patch.dict(os.environ, {"LOG_LEVEL": "debug"}):
            assert get_log_level() == logging.DEBUG

    def test_invalid_level_defaults_to_info(self) -> None:
        """Invalid log level should default to INFO."""
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            assert get_log_level() == logging.INFO


class TestGetLogFormat:
    """Tests for get_log_format function."""

    def test_default_is_text(self) -> None:
        """Default log format should be text."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LOG_FORMAT", None)
            assert get_log_format() == "text"

    def test_json_format(self) -> None:
        """LOG_FORMAT=json should return json."""
        with patch.dict(os.environ, {"LOG_FORMAT": "json"}):
            assert get_log_format() == "json"

    def test_text_format(self) -> None:
        """LOG_FORMAT=text should return text."""
        with patch.dict(os.environ, {"LOG_FORMAT": "text"}):
            assert get_log_format() == "text"

    def test_case_insensitive(self) -> None:
        """Log format should be case insensitive."""
        with patch.dict(os.environ, {"LOG_FORMAT": "JSON"}):
            assert get_log_format() == "json"

    def test_invalid_format_defaults_to_text(self) -> None:
        """Invalid log format should default to text."""
        with patch.dict(os.environ, {"LOG_FORMAT": "invalid"}):
            assert get_log_format() == "text"


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_formats_as_valid_json(self) -> None:
        """Output should be valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)  # Should not raise
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert "timestamp" in data

    def test_includes_source_for_debug(self) -> None:
        """Debug logs should include source location."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="/path/to/file.py",
            lineno=100,
            msg="Debug message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "source" in data
        assert data["source"]["line"] == 100
        assert data["source"]["file"] == "/path/to/file.py"

    def test_includes_source_for_error(self) -> None:
        """Error logs should include source location."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=50,
            msg="Error message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "source" in data
        assert data["source"]["line"] == 50

    def test_no_source_for_info(self) -> None:
        """Info logs should not include source location."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=50,
            msg="Info message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "source" not in data

    def test_formats_message_with_args(self) -> None:
        """Message arguments should be formatted."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Count: %d, Name: %s",
            args=(42, "test"),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "Count: 42, Name: test"


class TestTextFormatter:
    """Tests for text log formatter."""

    def test_formats_basic_message(self) -> None:
        """Basic message should be formatted correctly."""
        formatter = TextFormatter(use_colors=False)
        record = logging.LogRecord(
            name="loopengine.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "Test message" in output
        assert "INFO" in output
        assert "test" in output  # Shortened logger name

    def test_shortens_logger_name(self) -> None:
        """Logger names under loopengine should be shortened."""
        formatter = TextFormatter(use_colors=False)
        record = logging.LogRecord(
            name="loopengine.server.app",
            level=logging.INFO,
            pathname="app.py",
            lineno=1,
            msg="Message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "server.app" in output
        assert "loopengine.server.app" not in output

    def test_includes_source_for_debug(self) -> None:
        """Debug logs should include file:line."""
        formatter = TextFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="/path/to/test.py",
            lineno=99,
            msg="Debug",
            args=(),
            exc_info=None,
        )
        record.filename = "test.py"
        output = formatter.format(record)
        assert "test.py:99" in output


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configures_loopengine_logger(self) -> None:
        """Should configure the loopengine logger."""
        configure_logging(level=logging.DEBUG, format_type="text")
        logger = logging.getLogger("loopengine")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1

    def test_uses_json_formatter(self) -> None:
        """Should use JSON formatter when format_type is json."""
        configure_logging(level=logging.INFO, format_type="json")
        logger = logging.getLogger("loopengine")
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_uses_text_formatter(self) -> None:
        """Should use text formatter when format_type is text."""
        configure_logging(level=logging.INFO, format_type="text")
        logger = logging.getLogger("loopengine")
        assert isinstance(logger.handlers[0].formatter, TextFormatter)

    def test_reads_from_environment(self) -> None:
        """Should read level and format from environment."""
        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING", "LOG_FORMAT": "json"}):
            configure_logging()
            logger = logging.getLogger("loopengine")
            assert logger.level == logging.WARNING
            assert isinstance(logger.handlers[0].formatter, JSONFormatter)


class TestGetLogger:
    """Tests for get_logger convenience function."""

    def test_returns_logger(self) -> None:
        """Should return a logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_prefixes_loopengine(self) -> None:
        """Should prefix non-loopengine names with loopengine."""
        logger = get_logger("my_module")
        assert logger.name == "loopengine.my_module"

    def test_preserves_loopengine_prefix(self) -> None:
        """Should not double-prefix loopengine names."""
        logger = get_logger("loopengine.server")
        assert logger.name == "loopengine.server"


class TestIntegration:
    """Integration tests for logging."""

    def test_logging_to_stream(self) -> None:
        """Verify logs are written to stream."""
        # Create a string buffer to capture output
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(TextFormatter(use_colors=False))

        logger = logging.getLogger("loopengine.test_integration")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        logger.info("Integration test message")

        output = buffer.getvalue()
        assert "Integration test message" in output
        assert "INFO" in output

    def test_json_logging_to_stream(self) -> None:
        """Verify JSON logs are written correctly."""
        buffer = StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setFormatter(JSONFormatter())

        logger = logging.getLogger("loopengine.test_json_integration")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.warning("JSON test message")

        output = buffer.getvalue()
        data = json.loads(output.strip())
        assert data["message"] == "JSON test message"
        assert data["level"] == "WARNING"

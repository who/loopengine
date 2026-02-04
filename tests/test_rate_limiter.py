"""Tests for the rate limit handler module."""

import time
from unittest.mock import MagicMock

import pytest

from loopengine.behaviors.llm_client import BehaviorResponse, LLMQuery
from loopengine.behaviors.rate_limiter import (
    RateLimitConfig,
    RateLimitError,
    RateLimitEvent,
    RateLimitExhaustedError,
    RateLimitHandler,
    RateLimitStrategy,
    extract_retry_after,
    is_rate_limit_exception,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.strategy == RateLimitStrategy.RETRY_WITH_BACKOFF
        assert config.max_retries == 3
        assert config.initial_wait == 1.0
        assert config.max_wait == 60.0
        assert config.exponential_base == 2.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RateLimitConfig(
            strategy=RateLimitStrategy.FALLBACK_IMMEDIATELY,
            max_retries=5,
            initial_wait=2.0,
            max_wait=120.0,
            exponential_base=3.0,
        )
        assert config.strategy == RateLimitStrategy.FALLBACK_IMMEDIATELY
        assert config.max_retries == 5
        assert config.initial_wait == 2.0
        assert config.max_wait == 120.0
        assert config.exponential_base == 3.0

    def test_config_validation_max_retries(self) -> None:
        """Test max_retries validation bounds."""
        # Valid values
        RateLimitConfig(max_retries=1)
        RateLimitConfig(max_retries=10)

        # Invalid values
        with pytest.raises(ValueError):
            RateLimitConfig(max_retries=0)
        with pytest.raises(ValueError):
            RateLimitConfig(max_retries=11)


class TestRateLimitEvent:
    """Tests for RateLimitEvent model."""

    def test_default_event(self) -> None:
        """Test default event values."""
        event = RateLimitEvent(provider="claude")
        assert event.provider == "claude"
        assert event.retry_count == 0
        assert event.wait_time == 0.0
        assert event.resolved is False
        assert event.fallback_used is False
        assert event.timestamp > 0

    def test_event_with_values(self) -> None:
        """Test event with custom values."""
        event = RateLimitEvent(
            provider="openai",
            retry_count=3,
            wait_time=7.5,
            resolved=True,
            fallback_used=False,
        )
        assert event.provider == "openai"
        assert event.retry_count == 3
        assert event.wait_time == 7.5
        assert event.resolved is True


class TestRateLimitHandler:
    """Tests for RateLimitHandler class."""

    def test_init_with_defaults(self) -> None:
        """Test handler initialization with defaults."""
        handler = RateLimitHandler(provider="claude")
        assert handler.provider == "claude"
        assert handler.config.strategy == RateLimitStrategy.RETRY_WITH_BACKOFF
        assert len(handler.events) == 0

    def test_init_with_custom_config(self) -> None:
        """Test handler initialization with custom config."""
        config = RateLimitConfig(max_retries=5)
        handler = RateLimitHandler(provider="openai", config=config)
        assert handler.provider == "openai"
        assert handler.config.max_retries == 5

    def test_configure_updates_config(self) -> None:
        """Test configure method updates config."""
        handler = RateLimitHandler(provider="claude")
        new_config = RateLimitConfig(
            strategy=RateLimitStrategy.FALLBACK_IMMEDIATELY,
            max_retries=2,
        )
        handler.configure(new_config)
        assert handler.config.strategy == RateLimitStrategy.FALLBACK_IMMEDIATELY
        assert handler.config.max_retries == 2

    def test_set_fallback(self) -> None:
        """Test setting fallback function."""
        handler = RateLimitHandler(provider="claude")

        def fallback(query: LLMQuery) -> BehaviorResponse:
            return BehaviorResponse(action="idle", parameters={}, reasoning="fallback")

        handler.set_fallback(fallback)
        assert handler._fallback_fn is not None

    def test_execute_with_retry_success_no_retry(self) -> None:
        """Test successful execution without needing retries."""
        handler = RateLimitHandler(provider="claude")

        def success_fn() -> str:
            return "success"

        result = handler.execute_with_retry(success_fn)
        assert result == "success"
        assert len(handler.events) == 0

    def test_execute_with_retry_success_after_retries(self) -> None:
        """Test successful execution after retries."""
        config = RateLimitConfig(
            max_retries=3,
            initial_wait=0.01,  # Very short for tests
            max_wait=0.1,
        )
        handler = RateLimitHandler(provider="claude", config=config)

        call_count = 0

        def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("Rate limit", provider="claude")
            return "success"

        result = handler.execute_with_retry(
            fail_then_succeed,
            rate_limit_exceptions=(RateLimitError,),
        )
        assert result == "success"
        assert call_count == 2
        # Event logged for the retry
        assert len(handler.events) == 1
        assert handler.events[0].resolved is True

    def test_execute_with_retry_exhausted_no_fallback(self) -> None:
        """Test retries exhausted with no fallback raises error."""
        config = RateLimitConfig(
            max_retries=2,
            initial_wait=0.01,
            max_wait=0.1,
        )
        handler = RateLimitHandler(provider="claude", config=config)

        def always_fail() -> str:
            raise RateLimitError("Rate limit", provider="claude")

        with pytest.raises(RateLimitExhaustedError) as exc_info:
            handler.execute_with_retry(
                always_fail,
                rate_limit_exceptions=(RateLimitError,),
            )

        assert "exhausted" in str(exc_info.value).lower()
        assert exc_info.value.provider == "claude"
        # max_retries=2 means 2 total attempts (tenacity stop_after_attempt)
        assert exc_info.value.total_attempts == 2

    def test_execute_with_retry_exhausted_with_fallback(self) -> None:
        """Test retries exhausted with fallback returns fallback response."""
        config = RateLimitConfig(
            max_retries=2,
            initial_wait=0.01,
            max_wait=0.1,
        )
        handler = RateLimitHandler(provider="claude", config=config)

        def fallback(query: LLMQuery) -> BehaviorResponse:
            return BehaviorResponse(
                action="fallback_action",
                parameters={"source": "fallback"},
                reasoning="Using fallback due to rate limit",
            )

        handler.set_fallback(fallback)

        def always_fail(query: LLMQuery) -> BehaviorResponse:
            raise RateLimitError("Rate limit", provider="claude")

        query = LLMQuery(prompt="test")
        result = handler.execute_with_retry(
            always_fail,
            query,
            rate_limit_exceptions=(RateLimitError,),
        )

        assert result.action == "fallback_action"
        assert result.parameters == {"source": "fallback"}
        assert len(handler.events) == 1
        assert handler.events[0].fallback_used is True

    def test_execute_with_immediate_fallback_strategy(self) -> None:
        """Test immediate fallback strategy skips retries."""
        config = RateLimitConfig(
            strategy=RateLimitStrategy.FALLBACK_IMMEDIATELY,
            max_retries=5,  # Should not matter
        )
        handler = RateLimitHandler(provider="claude", config=config)

        def fallback(query: LLMQuery) -> BehaviorResponse:
            return BehaviorResponse(action="immediate_fallback", parameters={}, reasoning="")

        handler.set_fallback(fallback)

        call_count = 0

        def fail_on_first(query: LLMQuery) -> BehaviorResponse:
            nonlocal call_count
            call_count += 1
            raise RateLimitError("Rate limit", provider="claude")

        query = LLMQuery(prompt="test")
        result = handler.execute_with_retry(
            fail_on_first,
            query,
            rate_limit_exceptions=(RateLimitError,),
        )

        assert result.action == "immediate_fallback"
        assert call_count == 1  # Only one attempt, no retries

    def test_clear_events(self) -> None:
        """Test clearing events."""
        handler = RateLimitHandler(provider="claude")
        handler._events.append(RateLimitEvent(provider="claude"))
        handler._events.append(RateLimitEvent(provider="claude"))

        count = handler.clear_events()
        assert count == 2
        assert len(handler.events) == 0

    def test_get_stats_empty(self) -> None:
        """Test stats with no events."""
        handler = RateLimitHandler(provider="claude")
        stats = handler.get_stats()

        assert stats["total_events"] == 0
        assert stats["resolved_count"] == 0
        assert stats["fallback_count"] == 0
        assert stats["total_retries"] == 0
        assert stats["total_wait_time"] == 0.0

    def test_get_stats_with_events(self) -> None:
        """Test stats with events."""
        handler = RateLimitHandler(provider="claude")
        handler._events.append(
            RateLimitEvent(provider="claude", retry_count=2, wait_time=3.0, resolved=True)
        )
        handler._events.append(
            RateLimitEvent(provider="claude", retry_count=3, wait_time=5.0, fallback_used=True)
        )

        stats = handler.get_stats()
        assert stats["total_events"] == 2
        assert stats["resolved_count"] == 1
        assert stats["fallback_count"] == 1
        assert stats["total_retries"] == 5
        assert stats["total_wait_time"] == 8.0


class TestRateLimitExceptions:
    """Tests for RateLimitError and RateLimitExhaustedError."""

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError attributes."""
        original = ValueError("original")
        error = RateLimitError(
            "Rate limit hit",
            provider="claude",
            retry_after=30.0,
            original_error=original,
        )
        assert str(error) == "Rate limit hit"
        assert error.provider == "claude"
        assert error.retry_after == 30.0
        assert error.original_error is original

    def test_rate_limit_exhausted_error(self) -> None:
        """Test RateLimitExhaustedError attributes."""
        error = RateLimitExhaustedError(
            "Retries exhausted",
            provider="openai",
            total_attempts=4,
            total_wait_time=15.5,
        )
        assert str(error) == "Retries exhausted"
        assert error.provider == "openai"
        assert error.total_attempts == 4
        assert error.total_wait_time == 15.5


class TestIsRateLimitException:
    """Tests for is_rate_limit_exception helper."""

    def test_rate_limit_by_class_name(self) -> None:
        """Test detection by exception class name."""

        class RateLimitError(Exception):
            pass

        assert is_rate_limit_exception(RateLimitError("test"))

    def test_rate_limit_by_message(self) -> None:
        """Test detection by error message."""
        assert is_rate_limit_exception(Exception("rate limit exceeded"))
        assert is_rate_limit_exception(Exception("Rate Limit"))
        assert is_rate_limit_exception(Exception("rate_limit"))
        assert is_rate_limit_exception(Exception("Too many requests"))
        assert is_rate_limit_exception(Exception("HTTP 429"))
        assert is_rate_limit_exception(Exception("quota exceeded"))
        assert is_rate_limit_exception(Exception("requests per minute limit"))

    def test_not_rate_limit(self) -> None:
        """Test non-rate-limit exceptions."""
        assert not is_rate_limit_exception(Exception("timeout"))
        assert not is_rate_limit_exception(Exception("connection error"))
        assert not is_rate_limit_exception(ValueError("invalid input"))


class TestExtractRetryAfter:
    """Tests for extract_retry_after helper."""

    def test_extract_from_attribute(self) -> None:
        """Test extraction from retry_after attribute."""
        error = MagicMock()
        error.retry_after = 30
        assert extract_retry_after(error) == 30.0

    def test_extract_from_response_headers(self) -> None:
        """Test extraction from response headers."""
        error = MagicMock()
        error.retry_after = None
        error.response = MagicMock()
        error.response.headers = {"retry-after": "45"}
        assert extract_retry_after(error) == 45.0

    def test_extract_from_message(self) -> None:
        """Test extraction from error message."""
        error = Exception("Rate limit, retry after 60 seconds")
        assert extract_retry_after(error) == 60.0

        error2 = Exception("Please retry in 30.5 sec")
        assert extract_retry_after(error2) == 30.5

    def test_no_retry_after(self) -> None:
        """Test when no retry-after info available."""
        error = Exception("Generic rate limit error")
        assert extract_retry_after(error) is None


class TestRateLimitHandlerIntegration:
    """Integration tests for rate limit handler with mocked LLM calls."""

    def test_successful_query_no_rate_limit(self) -> None:
        """Test normal operation without rate limits."""
        handler = RateLimitHandler(provider="claude")

        def mock_query(query: LLMQuery) -> BehaviorResponse:
            return BehaviorResponse(
                action="move",
                parameters={"x": 10},
                reasoning="test",
            )

        query = LLMQuery(prompt="test")
        result = handler.execute_with_retry(mock_query, query)

        assert result.action == "move"
        assert len(handler.events) == 0

    def test_rate_limit_recovery(self) -> None:
        """Test recovery from rate limit within retry window."""
        config = RateLimitConfig(
            max_retries=3,
            initial_wait=0.01,
            max_wait=0.1,
        )
        handler = RateLimitHandler(provider="claude", config=config)

        attempts = []

        def mock_query(query: LLMQuery) -> BehaviorResponse:
            attempts.append(time.time())
            if len(attempts) < 3:
                raise RateLimitError("Rate limit", provider="claude")
            return BehaviorResponse(
                action="success_after_retry",
                parameters={},
                reasoning="recovered",
            )

        query = LLMQuery(prompt="test")
        result = handler.execute_with_retry(
            mock_query,
            query,
            rate_limit_exceptions=(RateLimitError,),
        )

        assert result.action == "success_after_retry"
        assert len(attempts) == 3
        # Verify exponential backoff happened (some delay between attempts)
        for i in range(1, len(attempts)):
            assert attempts[i] > attempts[i - 1]

    def test_graceful_degradation_to_fallback(self) -> None:
        """Test graceful degradation to fallback behavior."""
        config = RateLimitConfig(
            max_retries=2,
            initial_wait=0.01,
            max_wait=0.1,
        )
        handler = RateLimitHandler(provider="openai", config=config)

        def fallback(query: LLMQuery) -> BehaviorResponse:
            return BehaviorResponse(
                action="idle",
                parameters={},
                reasoning="Graceful degradation",
                metadata={"fallback": True, "fallback_reason": "rate_limit"},
            )

        handler.set_fallback(fallback)

        def always_rate_limited(query: LLMQuery) -> BehaviorResponse:
            raise RateLimitError("Rate limited", provider="openai")

        query = LLMQuery(prompt="test")
        result = handler.execute_with_retry(
            always_rate_limited,
            query,
            rate_limit_exceptions=(RateLimitError,),
        )

        assert result.action == "idle"
        assert result.metadata.get("fallback") is True
        stats = handler.get_stats()
        assert stats["fallback_count"] == 1


class TestRateLimitImports:
    """Tests for module imports."""

    def test_import_from_behaviors(self) -> None:
        """Test imports from behaviors package."""
        from loopengine.behaviors import (
            RateLimitConfig,
            RateLimitError,
            RateLimitEvent,
            RateLimitExhaustedError,
            RateLimitHandler,
            RateLimitStrategy,
            extract_retry_after,
            is_rate_limit_exception,
        )

        assert RateLimitHandler is not None
        assert RateLimitConfig is not None
        assert RateLimitStrategy is not None
        assert RateLimitError is not None
        assert RateLimitExhaustedError is not None
        assert RateLimitEvent is not None
        assert is_rate_limit_exception is not None
        assert extract_retry_after is not None

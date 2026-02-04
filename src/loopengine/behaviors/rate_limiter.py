"""Rate limit handling for LLM provider clients.

This module provides graceful rate limit handling with exponential backoff,
configurable retry strategies, and fallback to cached/default behaviors.
Implements NFR-003 requirements.
"""

import logging
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, Field
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from loopengine.behaviors.llm_client import BehaviorResponse, LLMQuery

logger = logging.getLogger(__name__)


class RateLimitStrategy(StrEnum):
    """Available rate limit handling strategies."""

    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_IMMEDIATELY = "fallback_immediately"
    QUEUE_REQUEST = "queue_request"


class RateLimitEvent(BaseModel):
    """Details of a rate limit event for logging and monitoring.

    Attributes:
        provider: The LLM provider that triggered the rate limit.
        timestamp: Unix timestamp of the event.
        retry_count: Number of retries attempted.
        wait_time: Time waited between retries in seconds.
        resolved: Whether the rate limit was resolved by retry.
        fallback_used: Whether fallback behavior was used.
    """

    provider: str = Field(description="LLM provider name")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    wait_time: float = Field(default=0.0, description="Total wait time in seconds")
    resolved: bool = Field(default=False, description="Whether retry succeeded")
    fallback_used: bool = Field(default=False, description="Whether fallback was used")


class RateLimitConfig(BaseModel):
    """Configuration for rate limit handling.

    Attributes:
        strategy: The handling strategy to use.
        max_retries: Maximum number of retry attempts (1-10).
        initial_wait: Initial wait time in seconds before first retry.
        max_wait: Maximum wait time between retries.
        exponential_base: Base for exponential backoff calculation.
    """

    strategy: RateLimitStrategy = Field(
        default=RateLimitStrategy.RETRY_WITH_BACKOFF,
        description="Rate limit handling strategy",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts",
    )
    initial_wait: float = Field(
        default=1.0,
        ge=0.001,
        le=60.0,
        description="Initial wait time in seconds",
    )
    max_wait: float = Field(
        default=60.0,
        ge=0.01,
        le=300.0,
        description="Maximum wait time between retries",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Exponential backoff base",
    )


T = TypeVar("T")


class RateLimitError(Exception):
    """Exception indicating a rate limit was hit.

    Attributes:
        provider: The provider that triggered the rate limit.
        retry_after: Suggested retry delay from provider (if available).
        original_error: The original exception from the provider SDK.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: float | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retry_after = retry_after
        self.original_error = original_error


class RateLimitExhaustedError(Exception):
    """Exception when all retries are exhausted.

    Attributes:
        provider: The provider that triggered the rate limit.
        total_attempts: Total number of attempts made.
        total_wait_time: Total time spent waiting.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        total_attempts: int,
        total_wait_time: float,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.total_attempts = total_attempts
        self.total_wait_time = total_wait_time


class RateLimitHandler:
    """Handles rate limiting with exponential backoff and fallback support.

    Provides configurable retry logic using tenacity, logging of rate limit
    events, and integration with the fallback behavior system.

    Example:
        >>> handler = RateLimitHandler(provider="claude")
        >>> result = handler.execute_with_retry(client.query, query)
    """

    def __init__(
        self,
        provider: str,
        config: RateLimitConfig | None = None,
        fallback_fn: Callable[[LLMQuery], BehaviorResponse] | None = None,
    ) -> None:
        """Initialize the rate limit handler.

        Args:
            provider: Name of the LLM provider (for logging).
            config: Rate limit configuration. Uses defaults if not provided.
            fallback_fn: Optional function to call when rate limits exhaust.
        """
        self._provider = provider
        self._config = config or RateLimitConfig()
        self._fallback_fn = fallback_fn
        self._events: list[RateLimitEvent] = []

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self._provider

    @property
    def config(self) -> RateLimitConfig:
        """Get the current configuration."""
        return self._config

    @property
    def events(self) -> list[RateLimitEvent]:
        """Get list of rate limit events."""
        return list(self._events)

    def configure(self, config: RateLimitConfig) -> None:
        """Update the rate limit configuration.

        Args:
            config: New configuration to apply.
        """
        self._config = config
        logger.debug(
            "Updated rate limit config for %s: strategy=%s, max_retries=%d",
            self._provider,
            config.strategy.value,
            config.max_retries,
        )

    def set_fallback(self, fallback_fn: Callable[[LLMQuery], BehaviorResponse]) -> None:
        """Set the fallback function for exhausted retries.

        Args:
            fallback_fn: Function that takes LLMQuery and returns BehaviorResponse.
        """
        self._fallback_fn = fallback_fn

    def execute_with_retry(
        self,
        fn: Callable[..., T],
        *args: Any,
        rate_limit_exceptions: tuple[type[Exception], ...] = (RateLimitError,),
        **kwargs: Any,
    ) -> T:
        """Execute a function with rate limit retry logic.

        Uses tenacity for exponential backoff retry when rate limits are hit.
        Falls back to fallback_fn if all retries are exhausted and a fallback
        is configured.

        Args:
            fn: The function to execute (typically client.query).
            *args: Positional arguments to pass to fn.
            rate_limit_exceptions: Exception types to retry on.
            **kwargs: Keyword arguments to pass to fn.

        Returns:
            The result from fn, or fallback response if retries exhausted.

        Raises:
            RateLimitExhaustedError: If retries exhausted and no fallback.
        """
        if self._config.strategy == RateLimitStrategy.FALLBACK_IMMEDIATELY:
            return self._execute_with_immediate_fallback(fn, *args, **kwargs)

        event = RateLimitEvent(provider=self._provider)
        start_time = time.time()

        def before_sleep(retry_state: RetryCallState) -> None:
            """Log before sleeping between retries."""
            wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
            event.retry_count = retry_state.attempt_number
            logger.warning(
                "Rate limit hit for %s, attempt %d/%d, waiting %.1fs",
                self._provider,
                retry_state.attempt_number,
                self._config.max_retries,
                wait_time,
            )

        try:
            retryer = Retrying(
                stop=stop_after_attempt(self._config.max_retries),
                wait=wait_exponential(
                    multiplier=self._config.initial_wait,
                    max=self._config.max_wait,
                    exp_base=self._config.exponential_base,
                ),
                retry=retry_if_exception_type(rate_limit_exceptions),
                before_sleep=before_sleep,
                reraise=True,
            )

            result = retryer(fn, *args, **kwargs)

            # Successful after retries
            if event.retry_count > 0:
                event.wait_time = time.time() - start_time
                event.resolved = True
                self._log_event(event)

            return result

        except rate_limit_exceptions:
            # Retries exhausted
            event.wait_time = time.time() - start_time
            event.resolved = False
            self._log_event(event)

            return self._handle_exhausted_retries(event, *args, **kwargs)

    def _execute_with_immediate_fallback(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute with immediate fallback on rate limit (no retries).

        Args:
            fn: The function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result from fn or fallback.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if self._is_rate_limit_error(e):
                event = RateLimitEvent(
                    provider=self._provider,
                    retry_count=0,
                    fallback_used=True,
                )
                self._log_event(event)
                return self._handle_exhausted_retries(event, *args, **kwargs)
            raise

    def _handle_exhausted_retries(
        self,
        event: RateLimitEvent,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Handle exhausted retries by using fallback or raising.

        Args:
            event: The rate limit event details.
            *args: Original function arguments.
            **kwargs: Original keyword arguments.

        Returns:
            Fallback response if available.

        Raises:
            RateLimitExhaustedError: If no fallback is configured.
        """
        if self._fallback_fn is not None:
            # Extract LLMQuery from args if present
            query = None
            for arg in args:
                if isinstance(arg, LLMQuery):
                    query = arg
                    break

            if query is not None:
                logger.info(
                    "Using fallback behavior for %s after %d retries",
                    self._provider,
                    event.retry_count,
                )
                event.fallback_used = True
                return self._fallback_fn(query)

        raise RateLimitExhaustedError(
            f"Rate limit exhausted for {self._provider} after {event.retry_count} retries",
            provider=self._provider,
            total_attempts=event.retry_count + 1,
            total_wait_time=event.wait_time,
        )

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an exception is a rate limit error.

        Args:
            error: The exception to check.

        Returns:
            True if this is a rate limit error.
        """
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()

        return (
            "ratelimit" in error_type
            or "rate_limit" in error_type
            or "rate limit" in error_msg
            or "rate_limit" in error_msg
            or "429" in error_msg
        )

    def _log_event(self, event: RateLimitEvent) -> None:
        """Log and store a rate limit event.

        Args:
            event: The event to log.
        """
        self._events.append(event)

        log_msg = (
            f"Rate limit event: provider={event.provider}, "
            f"retries={event.retry_count}, "
            f"wait_time={event.wait_time:.2f}s, "
            f"resolved={event.resolved}, "
            f"fallback_used={event.fallback_used}"
        )

        if event.resolved:
            logger.info(log_msg)
        elif event.fallback_used:
            logger.warning(log_msg)
        else:
            logger.error(log_msg)

    def clear_events(self) -> int:
        """Clear stored rate limit events.

        Returns:
            Number of events cleared.
        """
        count = len(self._events)
        self._events.clear()
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about rate limit events.

        Returns:
            Dictionary with event statistics.
        """
        if not self._events:
            return {
                "total_events": 0,
                "resolved_count": 0,
                "fallback_count": 0,
                "total_retries": 0,
                "total_wait_time": 0.0,
            }

        return {
            "total_events": len(self._events),
            "resolved_count": sum(1 for e in self._events if e.resolved),
            "fallback_count": sum(1 for e in self._events if e.fallback_used),
            "total_retries": sum(e.retry_count for e in self._events),
            "total_wait_time": sum(e.wait_time for e in self._events),
        }


def is_rate_limit_exception(error: Exception) -> bool:
    """Check if an exception indicates a rate limit.

    This function checks various exception types and messages to identify
    rate limit errors from different LLM providers.

    Args:
        error: The exception to check.

    Returns:
        True if this appears to be a rate limit error.
    """
    error_type = type(error).__name__.lower()
    error_msg = str(error).lower()

    # Check type name
    if "ratelimit" in error_type or "rate_limit" in error_type:
        return True

    # Check error message
    rate_limit_phrases = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "quota exceeded",
        "requests per minute",
        "tokens per minute",
    ]

    return any(phrase in error_msg for phrase in rate_limit_phrases)


def extract_retry_after(error: Exception) -> float | None:
    """Extract retry-after hint from an exception if available.

    Args:
        error: The exception to extract from.

    Returns:
        Retry delay in seconds, or None if not available.
    """
    import re

    # Check for retry_after attribute (some SDKs provide this)
    if hasattr(error, "retry_after"):
        retry_after = error.retry_after
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

    # Try to extract from response headers
    if hasattr(error, "response"):
        response = error.response
        if hasattr(response, "headers"):
            headers = response.headers
            if "retry-after" in headers:
                try:
                    return float(headers["retry-after"])
                except (ValueError, TypeError):
                    pass

    # Parse from error message (common format: "retry after X seconds")
    error_msg = str(error).lower()

    # Try several patterns
    patterns = [
        r"retry.{0,10}?(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)",
        r"after\s+(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)",
        r"wait\s+(\d+(?:\.\d+)?)\s*(?:second|sec|s\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_msg)
        if match:
            return float(match.group(1))

    return None

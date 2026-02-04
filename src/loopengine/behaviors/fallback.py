"""Fallback behavior mechanism for LLM unavailability.

This module provides fallback behaviors when LLM is unavailable due to
network issues, rate limits, API errors, or timeouts. Covers FR-004.
"""

import logging
from collections.abc import Callable
from typing import Any

from loopengine.behaviors.llm_client import BehaviorResponse

logger = logging.getLogger(__name__)


class FallbackReason:
    """Constants for fallback activation reasons."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    NO_API_KEY = "no_api_key"
    UNKNOWN = "unknown"


class FallbackBehavior:
    """Provides fallback behaviors when LLM is unavailable.

    Manages default actions per agent type and optional behavior caching
    for graceful degradation during LLM unavailability.

    Example:
        >>> fallback = FallbackBehavior()
        >>> fallback.register_default("florist", "arrange_flowers")
        >>> response = fallback.get_fallback("florist", FallbackReason.TIMEOUT)
        >>> print(response.action)  # "arrange_flowers"
    """

    # Default fallback action for unknown agent types
    DEFAULT_ACTION = "idle"

    def __init__(self) -> None:
        """Initialize the fallback behavior manager."""
        # Default actions per agent type
        self._default_actions: dict[str, str] = {}

        # Default parameters per agent type (optional)
        self._default_parameters: dict[str, dict[str, Any]] = {}

        # Recent behavior cache: key -> BehaviorResponse
        self._behavior_cache: dict[str, BehaviorResponse] = {}

        # Custom fallback handlers per agent type
        self._custom_handlers: dict[str, Callable[[str, dict[str, Any]], BehaviorResponse]] = {}

    def register_default(
        self,
        agent_type: str,
        action: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a default action for an agent type.

        Args:
            agent_type: The type of agent (e.g., "florist", "customer").
            action: The default action to take when LLM is unavailable.
            parameters: Optional default parameters for the action.
        """
        self._default_actions[agent_type] = action
        if parameters:
            self._default_parameters[agent_type] = parameters
        logger.debug("Registered default action '%s' for agent type '%s'", action, agent_type)

    def register_handler(
        self,
        agent_type: str,
        handler: Callable[[str, dict[str, Any]], BehaviorResponse],
    ) -> None:
        """Register a custom fallback handler for an agent type.

        The handler receives the fallback reason and context, and returns
        a BehaviorResponse. This allows for more sophisticated fallback
        logic than simple default actions.

        Args:
            agent_type: The type of agent.
            handler: Callable that takes (reason, context) and returns BehaviorResponse.
        """
        self._custom_handlers[agent_type] = handler
        logger.debug("Registered custom fallback handler for agent type '%s'", agent_type)

    def cache_behavior(
        self,
        agent_type: str,
        agent_id: str,
        response: BehaviorResponse,
    ) -> None:
        """Cache a successful behavior response for potential reuse.

        Cached behaviors can be used as fallbacks before resorting to
        default actions.

        Args:
            agent_type: The type of agent.
            agent_id: Unique identifier for the agent instance.
            response: The successful behavior response to cache.
        """
        cache_key = f"{agent_type}:{agent_id}"
        self._behavior_cache[cache_key] = response
        logger.debug("Cached behavior for %s", cache_key)

    def get_cached_behavior(
        self,
        agent_type: str,
        agent_id: str,
    ) -> BehaviorResponse | None:
        """Get a cached behavior response if available.

        Args:
            agent_type: The type of agent.
            agent_id: Unique identifier for the agent instance.

        Returns:
            Cached BehaviorResponse or None if not found.
        """
        cache_key = f"{agent_type}:{agent_id}"
        return self._behavior_cache.get(cache_key)

    def clear_cache(self, agent_type: str | None = None) -> int:
        """Clear cached behaviors.

        Args:
            agent_type: If provided, only clear cache for this agent type.
                If None, clears all cached behaviors.

        Returns:
            Number of cache entries cleared.
        """
        if agent_type is None:
            count = len(self._behavior_cache)
            self._behavior_cache.clear()
            logger.info("Cleared all %d cached behaviors", count)
            return count

        keys_to_remove = [k for k in self._behavior_cache if k.startswith(f"{agent_type}:")]
        for key in keys_to_remove:
            del self._behavior_cache[key]
        count = len(keys_to_remove)
        logger.info("Cleared %d cached behaviors for agent type '%s'", count, agent_type)
        return count

    def get_fallback(
        self,
        agent_type: str,
        reason: str,
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
        use_cache: bool = True,
    ) -> BehaviorResponse:
        """Get a fallback behavior for an agent when LLM is unavailable.

        Attempts fallback in this order:
        1. Custom handler (if registered)
        2. Cached behavior (if use_cache=True and available)
        3. Default action for agent type
        4. Global default action ("idle")

        Args:
            agent_type: The type of agent requesting fallback.
            reason: The reason for fallback (from FallbackReason constants).
            context: Optional context dict for custom handlers.
            agent_id: Optional agent ID for cache lookup.
            use_cache: Whether to attempt using cached behaviors.

        Returns:
            BehaviorResponse with the fallback action.
        """
        context = context or {}

        logger.warning(
            "Activating fallback for agent_type='%s' reason='%s'",
            agent_type,
            reason,
        )

        # Try custom handler first
        if agent_type in self._custom_handlers:
            try:
                response = self._custom_handlers[agent_type](reason, context)
                logger.info(
                    "Custom fallback handler for '%s' returned action='%s'",
                    agent_type,
                    response.action,
                )
                return self._enrich_metadata(response, reason, "custom_handler")
            except Exception as e:
                logger.error(
                    "Custom fallback handler for '%s' failed: %s",
                    agent_type,
                    str(e),
                )

        # Try cached behavior
        if use_cache and agent_id:
            cached = self.get_cached_behavior(agent_type, agent_id)
            if cached:
                logger.info(
                    "Using cached behavior for '%s:%s' action='%s'",
                    agent_type,
                    agent_id,
                    cached.action,
                )
                return self._enrich_metadata(cached, reason, "cache")

        # Use default action for agent type
        action = self._default_actions.get(agent_type, self.DEFAULT_ACTION)
        parameters = self._default_parameters.get(agent_type, {})

        logger.info(
            "Using default fallback for '%s' action='%s'",
            agent_type,
            action,
        )

        return BehaviorResponse(
            action=action,
            parameters=parameters,
            reasoning=f"Fallback due to {reason}",
            metadata={
                "fallback": True,
                "fallback_reason": reason,
                "fallback_source": "default",
                "agent_type": agent_type,
            },
        )

    def _enrich_metadata(
        self,
        response: BehaviorResponse,
        reason: str,
        source: str,
    ) -> BehaviorResponse:
        """Enrich a response with fallback metadata.

        Args:
            response: The original response.
            reason: The fallback reason.
            source: The fallback source (cache, custom_handler, default).

        Returns:
            New BehaviorResponse with enriched metadata.
        """
        enriched_metadata = dict(response.metadata)
        enriched_metadata.update(
            {
                "fallback": True,
                "fallback_reason": reason,
                "fallback_source": source,
            }
        )
        return BehaviorResponse(
            action=response.action,
            parameters=response.parameters,
            reasoning=response.reasoning,
            metadata=enriched_metadata,
        )

    def get_default_action(self, agent_type: str) -> str:
        """Get the registered default action for an agent type.

        Args:
            agent_type: The type of agent.

        Returns:
            The default action, or DEFAULT_ACTION if not registered.
        """
        return self._default_actions.get(agent_type, self.DEFAULT_ACTION)

    @property
    def registered_agent_types(self) -> list[str]:
        """Get list of agent types with registered defaults."""
        return list(self._default_actions.keys())

    @property
    def cache_size(self) -> int:
        """Get the current number of cached behaviors."""
        return len(self._behavior_cache)


def classify_error(error: Exception) -> str:
    """Classify an exception into a fallback reason.

    Args:
        error: The exception that occurred.

    Returns:
        A FallbackReason constant string.
    """
    error_type = type(error).__name__.lower()
    error_msg = str(error).lower()

    # Check for timeout (handles "timeout", "timed out", "TimeoutError", etc.)
    if "timeout" in error_type or "timeout" in error_msg or "timed out" in error_msg:
        return FallbackReason.TIMEOUT

    # Check for rate limit
    if "ratelimit" in error_type or "rate limit" in error_msg or "rate_limit" in error_msg:
        return FallbackReason.RATE_LIMIT

    # Check for network errors
    if any(
        term in error_type or term in error_msg
        for term in ["connection", "network", "socket", "unreachable", "dns"]
    ):
        return FallbackReason.NETWORK_ERROR

    # Check for API key issues
    if any(
        term in error_msg
        for term in ["api key", "api_key", "apikey", "authentication", "unauthorized"]
    ):
        return FallbackReason.NO_API_KEY

    # Check for general API errors
    if "api" in error_type or "api" in error_msg:
        return FallbackReason.API_ERROR

    return FallbackReason.UNKNOWN

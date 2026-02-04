"""Tests for the fallback behavior mechanism."""

from loopengine.behaviors import (
    BehaviorResponse,
    FallbackBehavior,
    FallbackReason,
    classify_error,
)


class TestFallbackReason:
    """Tests for FallbackReason constants."""

    def test_reason_constants_exist(self) -> None:
        """Verify all expected reason constants exist."""
        assert FallbackReason.TIMEOUT == "timeout"
        assert FallbackReason.RATE_LIMIT == "rate_limit"
        assert FallbackReason.API_ERROR == "api_error"
        assert FallbackReason.NETWORK_ERROR == "network_error"
        assert FallbackReason.NO_API_KEY == "no_api_key"
        assert FallbackReason.UNKNOWN == "unknown"


class TestFallbackBehavior:
    """Tests for the FallbackBehavior class."""

    def test_init(self) -> None:
        """Test initialization creates empty state."""
        fb = FallbackBehavior()
        assert fb.registered_agent_types == []
        assert fb.cache_size == 0

    def test_register_default_action(self) -> None:
        """Test registering a default action for an agent type."""
        fb = FallbackBehavior()
        fb.register_default("florist", "arrange_flowers")

        assert "florist" in fb.registered_agent_types
        assert fb.get_default_action("florist") == "arrange_flowers"

    def test_register_default_with_parameters(self) -> None:
        """Test registering a default action with parameters."""
        fb = FallbackBehavior()
        fb.register_default("customer", "wait", {"patience": 0.5})

        response = fb.get_fallback("customer", FallbackReason.TIMEOUT)
        assert response.action == "wait"
        assert response.parameters == {"patience": 0.5}

    def test_get_default_action_unregistered(self) -> None:
        """Test getting default action for unregistered agent type."""
        fb = FallbackBehavior()
        assert fb.get_default_action("unknown_type") == FallbackBehavior.DEFAULT_ACTION

    def test_get_fallback_with_registered_default(self) -> None:
        """Test getting fallback uses registered default action."""
        fb = FallbackBehavior()
        fb.register_default("driver", "deliver_package")

        response = fb.get_fallback("driver", FallbackReason.RATE_LIMIT)

        assert response.action == "deliver_package"
        assert response.metadata["fallback"] is True
        assert response.metadata["fallback_reason"] == "rate_limit"
        assert response.metadata["fallback_source"] == "default"
        assert response.metadata["agent_type"] == "driver"

    def test_get_fallback_unregistered_agent_type(self) -> None:
        """Test fallback for unregistered agent type uses global default."""
        fb = FallbackBehavior()

        response = fb.get_fallback("unknown_agent", FallbackReason.API_ERROR)

        assert response.action == FallbackBehavior.DEFAULT_ACTION  # "idle"
        assert response.metadata["fallback"] is True

    def test_get_fallback_reasoning_includes_reason(self) -> None:
        """Test fallback reasoning includes the failure reason."""
        fb = FallbackBehavior()

        response = fb.get_fallback("agent", FallbackReason.NETWORK_ERROR)

        assert "network_error" in response.reasoning

    def test_cache_behavior(self) -> None:
        """Test caching a behavior response."""
        fb = FallbackBehavior()
        response = BehaviorResponse(
            action="make_sandwich",
            parameters={"type": "blt"},
            reasoning="Customer ordered BLT",
        )

        fb.cache_behavior("sandwich_maker", "agent-001", response)

        assert fb.cache_size == 1
        cached = fb.get_cached_behavior("sandwich_maker", "agent-001")
        assert cached is not None
        assert cached.action == "make_sandwich"

    def test_get_cached_behavior_not_found(self) -> None:
        """Test getting cached behavior when not found."""
        fb = FallbackBehavior()
        cached = fb.get_cached_behavior("type", "nonexistent")
        assert cached is None

    def test_get_fallback_uses_cache(self) -> None:
        """Test fallback prefers cached behavior over default."""
        fb = FallbackBehavior()
        fb.register_default("maker", "default_action")

        cached_response = BehaviorResponse(
            action="cached_action",
            parameters={"cached": True},
            reasoning="Previously successful",
        )
        fb.cache_behavior("maker", "agent-123", cached_response)

        response = fb.get_fallback(
            "maker",
            FallbackReason.TIMEOUT,
            agent_id="agent-123",
            use_cache=True,
        )

        assert response.action == "cached_action"
        assert response.metadata["fallback_source"] == "cache"

    def test_get_fallback_cache_disabled(self) -> None:
        """Test fallback skips cache when disabled."""
        fb = FallbackBehavior()
        fb.register_default("maker", "default_action")

        cached_response = BehaviorResponse(
            action="cached_action",
            parameters={},
            reasoning="Previously successful",
        )
        fb.cache_behavior("maker", "agent-123", cached_response)

        response = fb.get_fallback(
            "maker",
            FallbackReason.TIMEOUT,
            agent_id="agent-123",
            use_cache=False,
        )

        # Should use default, not cache
        assert response.action == "default_action"
        assert response.metadata["fallback_source"] == "default"

    def test_clear_cache_all(self) -> None:
        """Test clearing all cached behaviors."""
        fb = FallbackBehavior()
        response = BehaviorResponse(action="test", parameters={}, reasoning="")

        fb.cache_behavior("type1", "id1", response)
        fb.cache_behavior("type2", "id2", response)
        assert fb.cache_size == 2

        cleared = fb.clear_cache()

        assert cleared == 2
        assert fb.cache_size == 0

    def test_clear_cache_by_agent_type(self) -> None:
        """Test clearing cache for specific agent type."""
        fb = FallbackBehavior()
        response = BehaviorResponse(action="test", parameters={}, reasoning="")

        fb.cache_behavior("type1", "id1", response)
        fb.cache_behavior("type1", "id2", response)
        fb.cache_behavior("type2", "id3", response)
        assert fb.cache_size == 3

        cleared = fb.clear_cache("type1")

        assert cleared == 2
        assert fb.cache_size == 1
        assert fb.get_cached_behavior("type2", "id3") is not None

    def test_register_custom_handler(self) -> None:
        """Test registering and using a custom fallback handler."""
        fb = FallbackBehavior()

        def custom_handler(reason: str, context: dict) -> BehaviorResponse:
            return BehaviorResponse(
                action="custom_fallback",
                parameters={"reason": reason, "custom": True},
                reasoning=f"Custom handler for {reason}",
            )

        fb.register_handler("special_agent", custom_handler)

        response = fb.get_fallback("special_agent", FallbackReason.API_ERROR)

        assert response.action == "custom_fallback"
        assert response.parameters["custom"] is True
        assert response.metadata["fallback_source"] == "custom_handler"

    def test_custom_handler_with_context(self) -> None:
        """Test custom handler receives context."""
        fb = FallbackBehavior()
        received_context = {}

        def context_handler(reason: str, context: dict) -> BehaviorResponse:
            received_context.update(context)
            return BehaviorResponse(
                action="context_aware",
                parameters=context,
                reasoning="Used context",
            )

        fb.register_handler("ctx_agent", context_handler)

        fb.get_fallback(
            "ctx_agent",
            FallbackReason.TIMEOUT,
            context={"key": "value", "number": 42},
        )

        assert received_context == {"key": "value", "number": 42}

    def test_custom_handler_failure_falls_through(self) -> None:
        """Test that failing custom handler falls through to cache/default."""
        fb = FallbackBehavior()
        fb.register_default("failing_agent", "default_action")

        def failing_handler(reason: str, context: dict) -> BehaviorResponse:
            raise ValueError("Handler failed")

        fb.register_handler("failing_agent", failing_handler)

        response = fb.get_fallback("failing_agent", FallbackReason.API_ERROR)

        # Should fall through to default
        assert response.action == "default_action"
        assert response.metadata["fallback_source"] == "default"

    def test_fallback_order_handler_first(self) -> None:
        """Test fallback order: handler > cache > default."""
        fb = FallbackBehavior()

        # Register all three fallback sources
        fb.register_default("agent", "default_action")
        fb.cache_behavior(
            "agent",
            "id1",
            BehaviorResponse(action="cached_action", parameters={}, reasoning=""),
        )
        fb.register_handler(
            "agent",
            lambda r, c: BehaviorResponse(
                action="handler_action",
                parameters={},
                reasoning="",
            ),
        )

        response = fb.get_fallback("agent", FallbackReason.TIMEOUT, agent_id="id1")

        # Handler should win
        assert response.action == "handler_action"

    def test_multiple_agent_types(self) -> None:
        """Test registering defaults for multiple agent types."""
        fb = FallbackBehavior()

        fb.register_default("florist", "arrange_flowers")
        fb.register_default("driver", "wait_for_order")
        fb.register_default("customer", "browse_menu")

        assert len(fb.registered_agent_types) == 3
        assert fb.get_default_action("florist") == "arrange_flowers"
        assert fb.get_default_action("driver") == "wait_for_order"
        assert fb.get_default_action("customer") == "browse_menu"


class TestClassifyError:
    """Tests for the classify_error function."""

    def test_classify_timeout_by_type(self) -> None:
        """Test classifying timeout errors by exception type."""

        class TimeoutError(Exception):
            pass

        result = classify_error(TimeoutError("Request timed out"))
        assert result == FallbackReason.TIMEOUT

    def test_classify_timeout_by_message(self) -> None:
        """Test classifying timeout errors by message content."""
        error = Exception("The request timed out after 30 seconds")
        result = classify_error(error)
        assert result == FallbackReason.TIMEOUT

    def test_classify_rate_limit_by_type(self) -> None:
        """Test classifying rate limit errors by exception type."""

        class RateLimitError(Exception):
            pass

        result = classify_error(RateLimitError("Too many requests"))
        assert result == FallbackReason.RATE_LIMIT

    def test_classify_rate_limit_by_message(self) -> None:
        """Test classifying rate limit errors by message content."""
        error = Exception("Rate limit exceeded, please retry later")
        result = classify_error(error)
        assert result == FallbackReason.RATE_LIMIT

    def test_classify_network_error_connection(self) -> None:
        """Test classifying connection errors as network errors."""

        class ConnectionError(Exception):
            pass

        result = classify_error(ConnectionError("Failed to connect"))
        assert result == FallbackReason.NETWORK_ERROR

    def test_classify_network_error_by_message(self) -> None:
        """Test classifying network errors by message content."""
        error = Exception("Network unreachable")
        result = classify_error(error)
        assert result == FallbackReason.NETWORK_ERROR

    def test_classify_socket_error(self) -> None:
        """Test classifying socket errors as network errors."""
        error = Exception("Socket error: connection refused")
        result = classify_error(error)
        assert result == FallbackReason.NETWORK_ERROR

    def test_classify_dns_error(self) -> None:
        """Test classifying DNS errors as network errors."""
        error = Exception("DNS resolution failed")
        result = classify_error(error)
        assert result == FallbackReason.NETWORK_ERROR

    def test_classify_api_key_error(self) -> None:
        """Test classifying API key errors."""
        error = Exception("Invalid API key provided")
        result = classify_error(error)
        assert result == FallbackReason.NO_API_KEY

    def test_classify_authentication_error(self) -> None:
        """Test classifying authentication errors."""
        error = Exception("Authentication failed: unauthorized")
        result = classify_error(error)
        assert result == FallbackReason.NO_API_KEY

    def test_classify_generic_api_error(self) -> None:
        """Test classifying generic API errors."""

        class APIError(Exception):
            pass

        result = classify_error(APIError("Something went wrong"))
        assert result == FallbackReason.API_ERROR

    def test_classify_unknown_error(self) -> None:
        """Test classifying unknown errors."""
        error = Exception("Something completely unexpected happened")
        result = classify_error(error)
        assert result == FallbackReason.UNKNOWN

    def test_classify_case_insensitive(self) -> None:
        """Test that classification is case insensitive."""
        error = Exception("TIMEOUT occurred")
        result = classify_error(error)
        assert result == FallbackReason.TIMEOUT


class TestIntegration:
    """Integration tests for fallback behavior."""

    def test_fallback_flow_timeout(self) -> None:
        """Test complete fallback flow for LLM timeout."""
        fb = FallbackBehavior()
        fb.register_default("sandwich_maker", "wait_for_order")

        # Simulate LLM timeout
        error = Exception("Request timed out after 3 seconds")
        reason = classify_error(error)

        response = fb.get_fallback("sandwich_maker", reason)

        assert response.action == "wait_for_order"
        assert response.metadata["fallback_reason"] == FallbackReason.TIMEOUT

    def test_fallback_flow_rate_limit(self) -> None:
        """Test complete fallback flow for rate limit."""
        fb = FallbackBehavior()
        fb.register_default("customer", "browse")

        # Simulate rate limit error
        error = Exception("Rate limit exceeded")
        reason = classify_error(error)

        response = fb.get_fallback("customer", reason)

        assert response.action == "browse"
        assert response.metadata["fallback_reason"] == FallbackReason.RATE_LIMIT

    def test_fallback_flow_network_failure(self) -> None:
        """Test complete fallback flow for network failure."""
        fb = FallbackBehavior()
        fb.register_default("driver", "wait")

        # Simulate network failure
        error = Exception("Connection refused")
        reason = classify_error(error)

        response = fb.get_fallback("driver", reason)

        assert response.action == "wait"
        assert response.metadata["fallback_reason"] == FallbackReason.NETWORK_ERROR

    def test_simulation_continues_with_fallback(self) -> None:
        """Test that simulation can continue with fallback behaviors."""
        fb = FallbackBehavior()

        # Register defaults for multiple agent types in a simulation
        fb.register_default("sandwich_maker", "prepare_workspace")
        fb.register_default("customer", "wait_in_line")
        fb.register_default("cashier", "ready_register")

        # Simulate getting behaviors for all agents during LLM unavailability
        responses = []
        for agent_type in ["sandwich_maker", "customer", "cashier"]:
            response = fb.get_fallback(agent_type, FallbackReason.TIMEOUT)
            responses.append(response)

        # All agents should have fallback behaviors
        assert len(responses) == 3
        assert all(r.metadata["fallback"] is True for r in responses)
        actions = [r.action for r in responses]
        assert "prepare_workspace" in actions
        assert "wait_in_line" in actions
        assert "ready_register" in actions

    def test_cache_reuse_during_outage(self) -> None:
        """Test cached behaviors are reused during LLM outage."""
        fb = FallbackBehavior()

        # Cache a successful behavior from before the outage
        successful_behavior = BehaviorResponse(
            action="make_blt",
            parameters={"bread": "sourdough", "toppings": ["bacon", "lettuce", "tomato"]},
            reasoning="Customer requested BLT on sourdough",
        )
        fb.cache_behavior("sandwich_maker", "maker-001", successful_behavior)

        # Now LLM is unavailable - use cached behavior
        response = fb.get_fallback(
            "sandwich_maker",
            FallbackReason.API_ERROR,
            agent_id="maker-001",
        )

        assert response.action == "make_blt"
        assert response.parameters["bread"] == "sourdough"
        assert response.metadata["fallback_source"] == "cache"

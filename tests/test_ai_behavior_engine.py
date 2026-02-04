"""Tests for the AI behavior engine orchestrator."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from loopengine.behaviors.ai_behavior_engine import (
    AIBehaviorEngine,
    AIBehaviorEngineError,
)
from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.llm_client import BehaviorResponse, LLMClient, LLMQuery
from loopengine.behaviors.prompt_builder import AgentContext, DomainContext, PromptBuilder
from loopengine.behaviors.response_parser import ResponseParser


@pytest.fixture
def mock_config() -> LLMConfig:
    """Create a mock LLM config."""
    return LLMConfig(
        llm_provider=LLMProvider.CLAUDE,
        anthropic_api_key=SecretStr("test-api-key"),
        llm_max_tokens=500,
        llm_temperature=0.7,
        llm_timeout=30.0,
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.query.return_value = BehaviorResponse(
        action="make_sandwich",
        parameters={"type": "turkey"},
        reasoning="Customer ordered a turkey sandwich",
        metadata={"model": "claude-sonnet-4-20250514"},
    )
    return client


@pytest.fixture
def domain_context() -> DomainContext:
    """Create test domain context."""
    return DomainContext(
        domain_type="sandwich shop",
        domain_description="A small sandwich shop with custom orders",
    )


@pytest.fixture
def agent_context() -> AgentContext:
    """Create test agent context."""
    return AgentContext(
        agent_type="sandwich_maker",
        agent_role="Prepares sandwiches for customers",
    )


class TestAIBehaviorEngineInit:
    """Tests for AIBehaviorEngine initialization."""

    def test_init_with_defaults(self, mock_config: LLMConfig) -> None:
        """Test engine initializes with default components."""
        with patch("loopengine.behaviors.ai_behavior_engine.ClaudeClient"):
            with patch(
                "loopengine.behaviors.ai_behavior_engine.get_llm_config",
                return_value=mock_config,
            ):
                engine = AIBehaviorEngine(config=mock_config)

                assert engine._config == mock_config
                assert isinstance(engine._prompt_builder, PromptBuilder)
                assert isinstance(engine._response_parser, ResponseParser)

    def test_init_with_injected_dependencies(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test engine initializes with injected dependencies."""
        prompt_builder = PromptBuilder()
        response_parser = ResponseParser()

        engine = AIBehaviorEngine(
            llm_client=mock_llm_client,
            prompt_builder=prompt_builder,
            response_parser=response_parser,
            config=mock_config,
        )

        assert engine._llm_client == mock_llm_client
        assert engine._prompt_builder == prompt_builder
        assert engine._response_parser == response_parser

    def test_init_metrics_start_at_zero(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test metrics are initialized to zero."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        metrics = engine.metrics
        assert metrics["total_queries"] == 0
        assert metrics["total_latency_ms"] == 0.0
        assert metrics["avg_latency_ms"] == 0.0


class TestAIBehaviorEngineClientCreation:
    """Tests for LLM client creation."""

    def test_creates_claude_client_for_claude_provider(self, mock_config: LLMConfig) -> None:
        """Test Claude client is created for claude provider."""
        with patch("loopengine.behaviors.ai_behavior_engine.ClaudeClient") as mock_claude:
            mock_claude.return_value = MagicMock(spec=LLMClient)

            engine = AIBehaviorEngine(config=mock_config)

            mock_claude.assert_called_once_with(mock_config)
            assert engine._llm_client is not None

    def test_raises_error_for_openai_provider(self) -> None:
        """Test error raised for unimplemented OpenAI provider."""
        config = LLMConfig(
            llm_provider=LLMProvider.OPENAI,
            openai_api_key=SecretStr("test-key"),
        )

        with pytest.raises(AIBehaviorEngineError) as exc_info:
            AIBehaviorEngine(config=config)

        assert "not yet implemented" in str(exc_info.value)

    def test_raises_error_for_ollama_provider(self) -> None:
        """Test error raised for unimplemented Ollama provider."""
        config = LLMConfig(
            llm_provider=LLMProvider.OLLAMA,
        )

        with pytest.raises(AIBehaviorEngineError) as exc_info:
            AIBehaviorEngine(config=config)

        assert "not yet implemented" in str(exc_info.value)


class TestAIBehaviorEngineGenerateBehavior:
    """Tests for generate_behavior method."""

    def test_generate_behavior_returns_response(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
        domain_context: DomainContext,
        agent_context: AgentContext,
    ) -> None:
        """Test generate_behavior returns a BehaviorResponse."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        response = engine.generate_behavior(domain_context, agent_context)

        assert isinstance(response, BehaviorResponse)
        assert response.action == "make_sandwich"
        assert response.parameters == {"type": "turkey"}

    def test_generate_behavior_includes_context(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
        domain_context: DomainContext,
        agent_context: AgentContext,
    ) -> None:
        """Test generate_behavior passes context to LLM query."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)
        context = {"pending_orders": 3, "current_task": None}

        engine.generate_behavior(domain_context, agent_context, context)

        # Verify LLM client was called
        mock_llm_client.query.assert_called_once()
        call_args = mock_llm_client.query.call_args
        query: LLMQuery = call_args[0][0]
        assert query.context == context

    def test_generate_behavior_enriches_metadata_with_latency(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
        domain_context: DomainContext,
        agent_context: AgentContext,
    ) -> None:
        """Test response metadata includes latency_ms."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        response = engine.generate_behavior(domain_context, agent_context)

        assert "latency_ms" in response.metadata
        assert isinstance(response.metadata["latency_ms"], float)
        assert response.metadata["latency_ms"] >= 0

    def test_generate_behavior_enriches_metadata_with_provider(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
        domain_context: DomainContext,
        agent_context: AgentContext,
    ) -> None:
        """Test response metadata includes provider."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        response = engine.generate_behavior(domain_context, agent_context)

        assert "provider" in response.metadata
        assert response.metadata["provider"] == "claude"

    def test_generate_behavior_updates_metrics(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
        domain_context: DomainContext,
        agent_context: AgentContext,
    ) -> None:
        """Test generate_behavior updates metrics."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        engine.generate_behavior(domain_context, agent_context)
        engine.generate_behavior(domain_context, agent_context)

        metrics = engine.metrics
        assert metrics["total_queries"] == 2
        assert metrics["total_latency_ms"] > 0
        assert metrics["avg_latency_ms"] > 0

    def test_generate_behavior_raises_on_llm_error(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
        domain_context: DomainContext,
        agent_context: AgentContext,
    ) -> None:
        """Test generate_behavior raises AIBehaviorEngineError on LLM failure."""
        mock_llm_client.query.side_effect = Exception("LLM query failed")
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(AIBehaviorEngineError) as exc_info:
            engine.generate_behavior(domain_context, agent_context)

        assert "Failed to generate behavior" in str(exc_info.value)


class TestAIBehaviorEngineThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_generate_behavior_calls(
        self,
        mock_config: LLMConfig,
        domain_context: DomainContext,
        agent_context: AgentContext,
    ) -> None:
        """Test concurrent generate_behavior calls are thread-safe."""
        call_count = 0
        call_lock = threading.Lock()

        def mock_query(query: LLMQuery) -> BehaviorResponse:
            nonlocal call_count
            with call_lock:
                call_count += 1
            # Small delay to increase chance of race conditions
            time.sleep(0.01)
            return BehaviorResponse(
                action="idle",
                parameters={},
                reasoning="Test",
                metadata={},
            )

        mock_client = MagicMock(spec=LLMClient)
        mock_client.query.side_effect = mock_query

        engine = AIBehaviorEngine(llm_client=mock_client, config=mock_config)

        threads = []
        results = []
        errors = []

        def worker() -> None:
            try:
                response = engine.generate_behavior(domain_context, agent_context)
                results.append(response)
            except Exception as e:
                errors.append(e)

        # Launch 10 concurrent threads
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0
        assert len(results) == 10

        # Verify metrics were updated correctly
        metrics = engine.metrics
        assert metrics["total_queries"] == 10

    def test_metrics_access_thread_safe(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test metrics access is thread-safe during updates."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)
        domain = DomainContext(domain_type="test")
        agent = AgentContext(agent_type="test")

        errors = []

        def writer() -> None:
            try:
                for _ in range(5):
                    engine.generate_behavior(domain, agent)
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(10):
                    _ = engine.metrics
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestAIBehaviorEngineParseRawResponse:
    """Tests for parse_raw_response method."""

    def test_parse_raw_response_valid_json(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test parsing valid JSON response."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        raw = '{"action": "wait", "parameters": {}, "reasoning": "Waiting for customer"}'
        response = engine.parse_raw_response(raw)

        assert response.action == "wait"
        assert response.reasoning == "Waiting for customer"

    def test_parse_raw_response_invalid_json(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test parsing invalid JSON returns fallback."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        raw = "This is not JSON"
        response = engine.parse_raw_response(raw)

        assert response.action == "idle"
        assert "parse_error" in response.metadata


class TestAIBehaviorEngineMetrics:
    """Tests for metrics functionality."""

    def test_metrics_returns_correct_values(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test metrics returns correct snapshot."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)
        domain = DomainContext(domain_type="test")
        agent = AgentContext(agent_type="test")

        engine.generate_behavior(domain, agent)
        engine.generate_behavior(domain, agent)

        metrics = engine.metrics
        assert metrics["total_queries"] == 2
        assert metrics["provider"] == "claude"
        assert "total_latency_ms" in metrics
        assert "avg_latency_ms" in metrics

    def test_reset_metrics(self, mock_config: LLMConfig, mock_llm_client: MagicMock) -> None:
        """Test reset_metrics clears counters."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)
        domain = DomainContext(domain_type="test")
        agent = AgentContext(agent_type="test")

        engine.generate_behavior(domain, agent)
        assert engine.metrics["total_queries"] == 1

        engine.reset_metrics()

        metrics = engine.metrics
        assert metrics["total_queries"] == 0
        assert metrics["total_latency_ms"] == 0.0
        assert metrics["avg_latency_ms"] == 0.0


class TestAIBehaviorEngineProperties:
    """Tests for engine properties."""

    def test_provider_property(self, mock_config: LLMConfig, mock_llm_client: MagicMock) -> None:
        """Test provider property returns correct value."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        assert engine.provider == "claude"


class TestAIBehaviorEngineImport:
    """Tests for import and instantiation."""

    def test_import_from_behaviors_package(self) -> None:
        """Test engine can be imported from behaviors package."""
        from loopengine.behaviors import AIBehaviorEngine, AIBehaviorEngineError

        assert AIBehaviorEngine is not None
        assert AIBehaviorEngineError is not None

    def test_full_integration_with_mocked_llm(self, mock_config: LLMConfig) -> None:
        """Test full integration with mocked LLM client."""
        from loopengine.behaviors import (
            AgentContext,
            AIBehaviorEngine,
            DomainContext,
        )

        mock_client = MagicMock(spec=LLMClient)
        mock_client.query.return_value = BehaviorResponse(
            action="greet_customer",
            parameters={"greeting": "Welcome!"},
            reasoning="New customer arrived",
            metadata={},
        )

        engine = AIBehaviorEngine(llm_client=mock_client, config=mock_config)
        domain = DomainContext(
            domain_type="flower shop",
            domain_description="A charming flower shop",
        )
        agent = AgentContext(
            agent_type="florist",
            agent_role="Helps customers select flowers",
        )
        context = {"customers_waiting": 1, "current_task": None}

        response = engine.generate_behavior(domain, agent, context)

        assert response.action == "greet_customer"
        assert "latency_ms" in response.metadata
        assert response.metadata["provider"] == "claude"

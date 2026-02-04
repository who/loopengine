"""Tests for concurrent behavior generation (NFR-004: 50+ agents).

This module tests thread-safety, concurrent access, and performance
of the behavior engine under load.
"""

import threading
import time
from concurrent.futures import Future, wait
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from loopengine.behaviors.ai_behavior_engine import (
    AIBehaviorEngine,
    AIBehaviorEngineError,
)
from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.fallback import FallbackBehavior
from loopengine.behaviors.llm_client import BehaviorResponse, LLMClient, LLMQuery
from loopengine.behaviors.prompt_builder import AgentContext, DomainContext


@pytest.fixture
def mock_config() -> LLMConfig:
    """Create a mock LLM config with concurrency settings."""
    return LLMConfig(
        llm_provider=LLMProvider.CLAUDE,
        anthropic_api_key=SecretStr("test-api-key"),
        llm_max_tokens=500,
        llm_temperature=0.7,
        llm_timeout=30.0,
        max_concurrent_requests=50,
        request_queue_size=100,
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client with simulated latency."""
    client = MagicMock(spec=LLMClient)

    def mock_query(query: LLMQuery) -> BehaviorResponse:
        # Simulate LLM latency (10-50ms)
        time.sleep(0.01 + (hash(query.prompt) % 40) / 1000)
        return BehaviorResponse(
            action="test_action",
            parameters={"processed": True},
            reasoning="Test response",
            metadata={"model": "test-model"},
        )

    client.query.side_effect = mock_query
    return client


@pytest.fixture
def domain_context() -> DomainContext:
    """Create test domain context."""
    return DomainContext(
        domain_type="test_domain",
        domain_description="A test domain for concurrent behavior testing",
    )


def create_agent_context(agent_id: int) -> AgentContext:
    """Create an agent context for a given agent ID."""
    return AgentContext(
        agent_type=f"agent_type_{agent_id % 5}",
        agent_role=f"Test role for agent {agent_id}",
    )


class TestConcurrentRequestTracking:
    """Tests for concurrent request tracking metrics."""

    def test_concurrent_request_counter_increments(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Test that concurrent request counter is tracked."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)
        agent = create_agent_context(1)

        # Make a single request
        engine.generate_behavior(domain_context, agent)

        metrics = engine.metrics
        assert metrics["total_queries"] == 1
        # After completion, concurrent should be 0
        assert metrics["concurrent_requests"] == 0
        # Peak should be at least 1
        assert metrics["peak_concurrent_requests"] >= 1

    def test_peak_concurrent_requests_tracked(
        self, mock_config: LLMConfig, domain_context: DomainContext
    ) -> None:
        """Test that peak concurrent requests is tracked accurately."""
        # Create a slow mock client to ensure overlap
        call_count = 0
        lock = threading.Lock()
        peak_observed = 0

        def slow_query(query: LLMQuery) -> BehaviorResponse:
            nonlocal call_count, peak_observed
            with lock:
                call_count += 1
                current = call_count
            time.sleep(0.05)  # 50ms to ensure overlap
            with lock:
                if current > peak_observed:
                    peak_observed = current
                call_count -= 1
            return BehaviorResponse(action="test", parameters={}, reasoning="", metadata={})

        client = MagicMock(spec=LLMClient)
        client.query.side_effect = slow_query

        engine = AIBehaviorEngine(llm_client=client, config=mock_config)

        # Launch multiple concurrent requests
        threads = []
        for i in range(10):
            agent = create_agent_context(i)
            t = threading.Thread(
                target=engine.generate_behavior,
                args=(domain_context, agent, None, f"agent_{i}"),
            )
            threads.append(t)

        # Start all threads nearly simultaneously
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        metrics = engine.metrics
        assert metrics["peak_concurrent_requests"] > 1
        assert metrics["total_queries"] == 10
        engine.shutdown()


class TestThreadPoolExecutor:
    """Tests for ThreadPoolExecutor-based concurrency."""

    def test_executor_created_lazily(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that executor is created only when needed."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)
        assert engine._executor is None

        # Trigger executor creation via async method
        domain = DomainContext(domain_type="test")
        agent = AgentContext(agent_type="test")
        future = engine.generate_behavior_async(domain, agent)
        future.result()

        assert engine._executor is not None
        engine.shutdown()

    def test_generate_behavior_async_returns_future(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Test that generate_behavior_async returns a Future."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)
        agent = create_agent_context(1)

        future = engine.generate_behavior_async(domain_context, agent)

        assert isinstance(future, Future)
        result = future.result()
        assert isinstance(result, BehaviorResponse)
        assert result.action == "test_action"
        engine.shutdown()

    def test_generate_behaviors_async_handles_multiple(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Test batch async request submission."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        requests = [
            (domain_context, create_agent_context(i), {"index": i}, f"agent_{i}") for i in range(10)
        ]

        futures = engine.generate_behaviors_async(requests)

        assert len(futures) == 10
        results = [f.result() for f in futures]
        assert all(isinstance(r, BehaviorResponse) for r in results)
        engine.shutdown()

    def test_generate_behaviors_batch_returns_all_results(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Test batch sync method returns all results."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        requests = [(domain_context, create_agent_context(i), None, f"agent_{i}") for i in range(5)]

        results = engine.generate_behaviors_batch(requests)

        assert len(results) == 5
        assert all(isinstance(r, BehaviorResponse) for r in results)
        engine.shutdown()

    def test_context_manager_shuts_down_executor(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Test that context manager properly shuts down executor."""
        with AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config) as engine:
            agent = create_agent_context(1)
            future = engine.generate_behavior_async(domain_context, agent)
            future.result()
            assert engine._executor is not None

        # After exit, executor should be None
        assert engine._executor is None


class TestFiftyAgentConcurrency:
    """Tests verifying NFR-004: 50+ concurrent agent decisions."""

    def test_50_concurrent_agents_succeed(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Test that 50 concurrent agent decisions all succeed."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        requests = [
            (domain_context, create_agent_context(i), {"agent_index": i}, f"agent_{i}")
            for i in range(50)
        ]

        results = engine.generate_behaviors_batch(requests, timeout=60.0)

        assert len(results) == 50
        assert all(isinstance(r, BehaviorResponse) for r in results)
        assert all(r.action == "test_action" for r in results)

        metrics = engine.metrics
        assert metrics["total_queries"] == 50
        engine.shutdown()

    def test_100_concurrent_agents_graceful_degradation(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Test that 100 concurrent requests are handled with graceful degradation."""
        # With max_concurrent_requests=50, the executor will queue extras
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        requests = [
            (domain_context, create_agent_context(i), None, f"agent_{i}") for i in range(100)
        ]

        results = engine.generate_behaviors_batch(requests, timeout=120.0)

        assert len(results) == 100
        assert all(isinstance(r, BehaviorResponse) for r in results)

        metrics = engine.metrics
        assert metrics["total_queries"] == 100
        engine.shutdown()

    def test_no_race_conditions_in_metrics(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Test that metrics are accurate under high concurrency."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        requests = [
            (domain_context, create_agent_context(i), None, f"agent_{i}") for i in range(50)
        ]

        _ = engine.generate_behaviors_batch(requests)

        metrics = engine.metrics
        assert metrics["total_queries"] == 50
        assert metrics["concurrent_requests"] == 0  # All completed
        assert metrics["peak_concurrent_requests"] > 0
        assert metrics["total_latency_ms"] > 0
        assert metrics["avg_latency_ms"] > 0
        engine.shutdown()


class TestThreadSafeCache:
    """Tests for thread-safe fallback cache."""

    def test_concurrent_cache_writes_no_corruption(self, mock_config: LLMConfig) -> None:
        """Test that concurrent cache writes don't cause corruption."""
        fallback = FallbackBehavior()
        errors = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(100):
                    response = BehaviorResponse(
                        action=f"action_{thread_id}_{i}",
                        parameters={"thread": thread_id, "index": i},
                        reasoning="Test",
                        metadata={},
                    )
                    fallback.cache_behavior(f"type_{thread_id}", f"agent_{i}", response)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Verify cache has expected entries
        assert fallback.cache_size == 1000  # 10 threads * 100 entries each

    def test_concurrent_cache_reads_no_errors(self, mock_config: LLMConfig) -> None:
        """Test that concurrent cache reads don't cause errors."""
        fallback = FallbackBehavior()

        # Pre-populate cache
        for i in range(100):
            response = BehaviorResponse(
                action=f"action_{i}",
                parameters={},
                reasoning="Test",
                metadata={},
            )
            fallback.cache_behavior("test_type", f"agent_{i}", response)

        errors = []

        def reader(thread_id: int) -> None:
            try:
                for i in range(100):
                    _ = fallback.get_cached_behavior("test_type", f"agent_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_read_write_no_errors(self, mock_config: LLMConfig) -> None:
        """Test that concurrent reads and writes don't cause errors."""
        fallback = FallbackBehavior()
        errors = []

        def writer() -> None:
            try:
                for i in range(50):
                    response = BehaviorResponse(
                        action=f"action_{i}",
                        parameters={},
                        reasoning="Test",
                        metadata={},
                    )
                    fallback.cache_behavior("test_type", f"agent_{i}", response)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for i in range(50):
                    _ = fallback.get_cached_behavior("test_type", f"agent_{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def clearer() -> None:
            try:
                for _ in range(5):
                    time.sleep(0.01)
                    fallback.clear_cache("test_type")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=clearer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestConcurrencyConfiguration:
    """Tests for concurrency configuration."""

    def test_max_concurrent_requests_config(self) -> None:
        """Test that max_concurrent_requests is configurable."""
        config = LLMConfig(
            llm_provider=LLMProvider.CLAUDE,
            anthropic_api_key=SecretStr("test-key"),
            max_concurrent_requests=100,
        )
        assert config.max_concurrent_requests == 100

    def test_default_max_concurrent_requests(self) -> None:
        """Test default max_concurrent_requests is 50."""
        config = LLMConfig(
            llm_provider=LLMProvider.CLAUDE,
            anthropic_api_key=SecretStr("test-key"),
        )
        assert config.max_concurrent_requests == 50

    def test_engine_uses_config_for_executor_size(
        self, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Test that engine uses config for thread pool size."""
        config = LLMConfig(
            llm_provider=LLMProvider.CLAUDE,
            anthropic_api_key=SecretStr("test-key"),
            max_concurrent_requests=25,
        )
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=config)

        # Trigger executor creation
        agent = create_agent_context(1)
        future = engine.generate_behavior_async(domain_context, agent)
        future.result()

        assert engine._executor._max_workers == 25
        engine.shutdown()


class TestErrorHandlingUnderConcurrency:
    """Tests for error handling under concurrent load."""

    def test_single_failure_doesnt_affect_others(
        self, mock_config: LLMConfig, domain_context: DomainContext
    ) -> None:
        """Test that one failed request doesn't affect others."""
        call_count = 0
        lock = threading.Lock()

        def flaky_query(query: LLMQuery) -> BehaviorResponse:
            nonlocal call_count
            with lock:
                call_count += 1
                current = call_count
            # Fail every 10th request
            if current % 10 == 0:
                raise Exception("Simulated failure")
            return BehaviorResponse(action="test", parameters={}, reasoning="", metadata={})

        client = MagicMock(spec=LLMClient)
        client.query.side_effect = flaky_query

        engine = AIBehaviorEngine(llm_client=client, config=mock_config)

        futures = []
        for i in range(50):
            agent = create_agent_context(i)
            futures.append(engine.generate_behavior_async(domain_context, agent))

        # Wait for all to complete
        wait(futures)

        # Count successes and failures
        successes = 0
        failures = 0
        for f in futures:
            try:
                f.result()
                successes += 1
            except AIBehaviorEngineError:
                failures += 1

        assert successes > 0
        assert failures == 5  # Every 10th of 50
        engine.shutdown()

    def test_batch_with_timeout_cancels_pending(
        self, mock_config: LLMConfig, domain_context: DomainContext
    ) -> None:
        """Test that batch with timeout cancels pending requests."""

        def slow_query(query: LLMQuery) -> BehaviorResponse:
            time.sleep(10)  # Very slow
            return BehaviorResponse(action="test", parameters={}, reasoning="", metadata={})

        client = MagicMock(spec=LLMClient)
        client.query.side_effect = slow_query

        engine = AIBehaviorEngine(llm_client=client, config=mock_config)

        requests = [
            (domain_context, create_agent_context(i), None, f"agent_{i}") for i in range(10)
        ]

        with pytest.raises(AIBehaviorEngineError) as exc_info:
            engine.generate_behaviors_batch(requests, timeout=0.1)

        assert "timed out" in str(exc_info.value).lower()
        engine.shutdown(wait=False)


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_benchmark_50_agents_latency(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Benchmark: 50 concurrent agents should complete reasonably fast."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        requests = [
            (domain_context, create_agent_context(i), None, f"agent_{i}") for i in range(50)
        ]

        start_time = time.perf_counter()
        results = engine.generate_behaviors_batch(requests, timeout=30.0)
        elapsed = time.perf_counter() - start_time

        assert len(results) == 50
        # With mocked 10-50ms latency and parallelism, should complete well under 30s
        # In practice, with 50 workers, should complete in ~50-100ms
        assert elapsed < 5.0, f"50 agents took {elapsed:.2f}s (expected < 5s)"

        metrics = engine.metrics
        # Log for visibility
        print(f"\nBenchmark: 50 agents in {elapsed:.3f}s")
        print(f"Peak concurrent: {metrics['peak_concurrent_requests']}")
        print(f"Avg latency: {metrics['avg_latency_ms']:.2f}ms")
        engine.shutdown()

    def test_benchmark_throughput(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock, domain_context: DomainContext
    ) -> None:
        """Benchmark: measure requests per second throughput."""
        engine = AIBehaviorEngine(llm_client=mock_llm_client, config=mock_config)

        requests = [
            (domain_context, create_agent_context(i), None, f"agent_{i}") for i in range(100)
        ]

        start_time = time.perf_counter()
        results = engine.generate_behaviors_batch(requests, timeout=60.0)
        elapsed = time.perf_counter() - start_time

        throughput = len(results) / elapsed
        print(f"\nThroughput benchmark: {throughput:.1f} requests/second")
        print(f"Total: {len(results)} requests in {elapsed:.3f}s")

        assert len(results) == 100
        engine.shutdown()

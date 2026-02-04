"""AI behavior engine orchestrator.

This module provides the main AIBehaviorEngine class that coordinates
domain context, prompt building, LLM querying, and response parsing
into a cohesive behavior generation flow.

Supports concurrent agent decisions (NFR-004: 50+ agents) via thread pool.
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from loopengine.behaviors.config import LLMConfig, LLMProvider, get_llm_config
from loopengine.behaviors.fallback import FallbackBehavior, FallbackReason
from loopengine.behaviors.latency_tracker import LatencyTracker
from loopengine.behaviors.llm_client import BehaviorResponse, LLMClient, LLMQuery
from loopengine.behaviors.prompt_builder import AgentContext, DomainContext, PromptBuilder
from loopengine.behaviors.providers.claude import ClaudeClient
from loopengine.behaviors.rate_limiter import (
    RateLimitExhaustedError,
    RateLimitHandler,
    is_rate_limit_exception,
)
from loopengine.behaviors.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class AIBehaviorEngineError(Exception):
    """Exception raised when behavior generation fails."""

    pass


class AIBehaviorEngine:
    """Orchestrates AI-driven behavior generation for simulation agents.

    Coordinates the flow: prompt_builder -> llm_client -> response_parser
    to generate agent behaviors from simulation context.

    Thread-safe for concurrent agent decisions through internal locking on
    shared resources. Supports 50+ concurrent agents via ThreadPoolExecutor
    (NFR-004 compliance).

    Example:
        >>> engine = AIBehaviorEngine()
        >>> domain = DomainContext(
        ...     domain_type="flower shop",
        ...     domain_description="A small florist"
        ... )
        >>> agent = AgentContext(
        ...     agent_type="florist",
        ...     agent_role="Prepares arrangements"
        ... )
        >>> context = {"pending_orders": 3}
        >>> response = engine.generate_behavior(domain, agent, context)
        >>> print(response.action)

        # Batch concurrent requests for 50+ agents
        >>> futures = engine.generate_behaviors_async([
        ...     (domain, agent1, context1, "agent1"),
        ...     (domain, agent2, context2, "agent2"),
        ... ])
        >>> responses = [f.result() for f in futures]
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: ResponseParser | None = None,
        config: LLMConfig | None = None,
        fallback_behavior: FallbackBehavior | None = None,
        latency_tracker: LatencyTracker | None = None,
    ) -> None:
        """Initialize the behavior engine with optional dependency injection.

        Args:
            llm_client: LLM client to use for queries. If not provided,
                creates one based on configuration.
            prompt_builder: Prompt builder instance. If not provided,
                creates a default one.
            response_parser: Response parser instance. If not provided,
                creates a default one.
            config: LLM configuration. If not provided, loads from environment.
            fallback_behavior: Fallback behavior manager. If not provided,
                creates a default one.
            latency_tracker: Latency tracker for monitoring. If not provided,
                creates a default one.
        """
        self._config = config or get_llm_config()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._response_parser = response_parser or ResponseParser()
        self._llm_client = llm_client or self._create_llm_client()
        self._fallback = fallback_behavior or FallbackBehavior()
        self._latency_tracker = latency_tracker or LatencyTracker()

        # Initialize rate limit handler
        self._rate_limit_handler = RateLimitHandler(
            provider=self._config.llm_provider.value,
            config=self._config.get_rate_limit_config(),
        )

        # Lock for thread-safe access to shared state
        self._lock = threading.Lock()

        # Thread pool for concurrent requests (NFR-004: 50+ agents)
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()

        # Metrics tracking
        self._total_queries = 0
        self._total_latency_ms = 0.0
        self._rate_limit_events = 0
        self._concurrent_requests = 0
        self._peak_concurrent_requests = 0

    def _create_llm_client(self) -> LLMClient:
        """Create an LLM client based on configuration.

        Returns:
            Configured LLM client for the selected provider.

        Raises:
            AIBehaviorEngineError: If provider is not supported.
        """
        provider = self._config.llm_provider

        if provider == LLMProvider.CLAUDE:
            return ClaudeClient(self._config)
        elif provider == LLMProvider.OPENAI:
            # OpenAI client not yet implemented
            raise AIBehaviorEngineError(f"Provider '{provider.value}' is not yet implemented")
        elif provider == LLMProvider.OLLAMA:
            # Ollama client not yet implemented
            raise AIBehaviorEngineError(f"Provider '{provider.value}' is not yet implemented")
        else:
            raise AIBehaviorEngineError(f"Unknown provider: {provider}")

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create the thread pool executor.

        Creates executor lazily on first use.

        Returns:
            ThreadPoolExecutor configured for max_concurrent_requests.
        """
        if self._executor is None:
            with self._executor_lock:
                # Double-check after acquiring lock
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self._config.max_concurrent_requests,
                        thread_name_prefix="behavior_worker",
                    )
                    logger.info(
                        "Created thread pool with %d workers for concurrent behavior generation",
                        self._config.max_concurrent_requests,
                    )
        return self._executor

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool executor.

        Call this when done using the engine to clean up resources.

        Args:
            wait: If True, wait for pending tasks to complete.
        """
        with self._executor_lock:
            if self._executor is not None:
                logger.info("Shutting down behavior engine thread pool")
                self._executor.shutdown(wait=wait)
                self._executor = None

    def __enter__(self) -> "AIBehaviorEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures executor shutdown."""
        self.shutdown(wait=True)

    def generate_behavior(
        self,
        domain: DomainContext,
        agent: AgentContext,
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> BehaviorResponse:
        """Generate a behavior decision for an agent.

        This is the main entry point for behavior generation. It coordinates:
        1. Building the prompt from domain, agent, and context
        2. Querying the LLM for a behavior decision (with rate limit handling)
        3. Parsing and validating the response
        4. Falling back to cached/default behavior on rate limit exhaustion

        Thread-safe for concurrent calls. Supports 50+ concurrent agent decisions.

        Args:
            domain: Domain configuration with type and description.
            agent: Agent information with type and role.
            context: Optional state context dict (current state, nearby agents, etc.)
            agent_id: Optional unique agent ID for cache lookup on fallback.

        Returns:
            BehaviorResponse with action, parameters, reasoning, and metadata
            including latency_ms and provider.

        Raises:
            AIBehaviorEngineError: If behavior generation fails fatally.
        """
        start_time = time.perf_counter()

        # Track concurrent requests for metrics
        with self._lock:
            self._concurrent_requests += 1
            if self._concurrent_requests > self._peak_concurrent_requests:
                self._peak_concurrent_requests = self._concurrent_requests

        try:
            # Build the prompt
            system_message, prompt = self._prompt_builder.build_full_prompt(domain, agent, context)

            # Create the query
            query = LLMQuery(
                prompt=prompt,
                system_message=system_message,
                context=context or {},
            )

            # Query the LLM with rate limit handling
            logger.debug(
                "Querying LLM for agent %s in domain %s",
                agent.agent_type,
                domain.domain_type,
            )

            # Set up fallback function for rate limit exhaustion
            def fallback_fn(q: LLMQuery) -> BehaviorResponse:
                return self._fallback.get_fallback(
                    agent_type=agent.agent_type,
                    reason=FallbackReason.RATE_LIMIT,
                    context=context,
                    agent_id=agent_id,
                )

            self._rate_limit_handler.set_fallback(fallback_fn)

            # Execute query with rate limit retry handling
            try:
                response = self._rate_limit_handler.execute_with_retry(
                    self._llm_client.query,
                    query,
                    rate_limit_exceptions=self._get_rate_limit_exceptions(),
                )
            except RateLimitExhaustedError:
                # Rate limit exhausted and no fallback configured, use our fallback
                with self._lock:
                    self._rate_limit_events += 1
                response = fallback_fn(query)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics (thread-safe)
            with self._lock:
                self._total_queries += 1
                self._total_latency_ms += latency_ms

            # Record latency with context for monitoring (NFR-001)
            self._latency_tracker.record(
                latency_ms=latency_ms,
                agent_type=agent.agent_type,
                domain_type=domain.domain_type,
                context={
                    "agent_id": agent_id,
                    "fallback": response.metadata.get("fallback", False),
                    "action": response.action,
                },
            )

            # Cache successful non-fallback responses
            if not response.metadata.get("fallback") and agent_id:
                self._fallback.cache_behavior(agent.agent_type, agent_id, response)

            # Enrich metadata with latency and provider info
            enriched_metadata = dict(response.metadata)
            enriched_metadata["latency_ms"] = round(latency_ms, 2)
            enriched_metadata["provider"] = self._config.llm_provider.value

            logger.info(
                "Generated behavior for %s: action=%s, latency=%.2fms%s",
                agent.agent_type,
                response.action,
                latency_ms,
                " (fallback)" if response.metadata.get("fallback") else "",
            )

            return BehaviorResponse(
                action=response.action,
                parameters=response.parameters,
                reasoning=response.reasoning,
                metadata=enriched_metadata,
            )

        except RateLimitExhaustedError as e:
            # This shouldn't happen with our fallback, but handle it anyway
            latency_ms = (time.perf_counter() - start_time) * 1000
            with self._lock:
                self._rate_limit_events += 1
            logger.error(
                "Rate limit exhausted for %s after %.2fms, %d attempts",
                agent.agent_type,
                latency_ms,
                e.total_attempts,
            )
            raise AIBehaviorEngineError(f"Rate limit exhausted: {e}") from e

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Check if this is a rate limit error that slipped through
            if is_rate_limit_exception(e):
                with self._lock:
                    self._rate_limit_events += 1
                logger.warning(
                    "Unexpected rate limit error for %s, using fallback",
                    agent.agent_type,
                )
                return self._fallback.get_fallback(
                    agent_type=agent.agent_type,
                    reason=FallbackReason.RATE_LIMIT,
                    context=context,
                    agent_id=agent_id,
                )

            logger.error(
                "Behavior generation failed for %s after %.2fms: %s",
                agent.agent_type,
                latency_ms,
                str(e),
            )
            raise AIBehaviorEngineError(f"Failed to generate behavior: {e}") from e

        finally:
            # Always decrement concurrent request count
            with self._lock:
                self._concurrent_requests -= 1

    def _get_rate_limit_exceptions(self) -> tuple[type[Exception], ...]:
        """Get the exception types that indicate rate limiting for the current provider.

        Returns:
            Tuple of exception types to retry on.
        """
        # Import provider-specific exceptions
        from anthropic import RateLimitError as AnthropicRateLimitError
        from openai import RateLimitError as OpenAIRateLimitError

        return (AnthropicRateLimitError, OpenAIRateLimitError)

    def generate_behavior_async(
        self,
        domain: DomainContext,
        agent: AgentContext,
        context: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> Future[BehaviorResponse]:
        """Submit a behavior generation request to the thread pool.

        Non-blocking. Returns a Future that will contain the result.

        Args:
            domain: Domain configuration with type and description.
            agent: Agent information with type and role.
            context: Optional state context dict.
            agent_id: Optional unique agent ID for cache lookup on fallback.

        Returns:
            Future that will contain the BehaviorResponse when complete.
        """
        executor = self._get_executor()
        return executor.submit(self.generate_behavior, domain, agent, context, agent_id)

    def generate_behaviors_async(
        self,
        requests: list[tuple[DomainContext, AgentContext, dict[str, Any] | None, str | None]],
    ) -> list[Future[BehaviorResponse]]:
        """Submit multiple behavior generation requests concurrently.

        Designed for NFR-004: supports 50+ concurrent agent decisions.

        Args:
            requests: List of tuples (domain, agent, context, agent_id).

        Returns:
            List of Futures in the same order as requests.

        Example:
            >>> futures = engine.generate_behaviors_async([
            ...     (domain, agent1, ctx1, "agent1"),
            ...     (domain, agent2, ctx2, "agent2"),
            ... ])
            >>> responses = [f.result() for f in futures]
        """
        executor = self._get_executor()
        futures: list[Future[BehaviorResponse]] = []

        for domain, agent, context, agent_id in requests:
            future = executor.submit(self.generate_behavior, domain, agent, context, agent_id)
            futures.append(future)

        logger.info(
            "Submitted %d concurrent behavior requests (max workers: %d)",
            len(futures),
            self._config.max_concurrent_requests,
        )
        return futures

    def generate_behaviors_batch(
        self,
        requests: list[tuple[DomainContext, AgentContext, dict[str, Any] | None, str | None]],
        timeout: float | None = None,
    ) -> list[BehaviorResponse]:
        """Generate behaviors for multiple agents concurrently, waiting for all to complete.

        Convenience method that submits all requests and waits for results.

        Args:
            requests: List of tuples (domain, agent, context, agent_id).
            timeout: Maximum time to wait for all results (seconds). None = no timeout.

        Returns:
            List of BehaviorResponses in the same order as requests.

        Raises:
            AIBehaviorEngineError: If any request fails or times out.
        """
        from concurrent.futures import TimeoutError, wait

        futures = self.generate_behaviors_async(requests)

        try:
            # Wait for all futures to complete
            _done, not_done = wait(futures, timeout=timeout)

            if not_done:
                # Cancel any pending futures and raise error
                for f in not_done:
                    f.cancel()
                incomplete = len(not_done)
                total = len(futures)
                raise AIBehaviorEngineError(
                    f"Batch request timed out: {incomplete} of {total} requests incomplete"
                )

            # Collect results in original order
            results: list[BehaviorResponse] = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    raise AIBehaviorEngineError(f"Batch request failed: {e}") from e

            return results

        except TimeoutError as e:
            raise AIBehaviorEngineError(f"Batch request timed out after {timeout}s") from e

    def parse_raw_response(self, raw_response: str) -> BehaviorResponse:
        """Parse a raw LLM response into a BehaviorResponse.

        Useful when working with responses from external LLM calls
        that need validation.

        Args:
            raw_response: Raw text response from an LLM.

        Returns:
            Parsed and validated BehaviorResponse.
        """
        return self._response_parser.parse(raw_response)

    @property
    def provider(self) -> str:
        """Get the current LLM provider name."""
        return self._config.llm_provider.value

    @property
    def metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot.

        Thread-safe access to metrics.

        Returns:
            Dict with total_queries, total_latency_ms, avg_latency_ms,
            rate_limit_events, concurrency stats, latency percentiles,
            and rate limit handler stats.
        """
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._total_queries if self._total_queries > 0 else 0.0
            )
            rate_limit_stats = self._rate_limit_handler.get_stats()
            latency_stats = self._latency_tracker.get_stats()
            return {
                "total_queries": self._total_queries,
                "total_latency_ms": round(self._total_latency_ms, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "provider": self._config.llm_provider.value,
                "rate_limit_events": self._rate_limit_events,
                "concurrent_requests": self._concurrent_requests,
                "peak_concurrent_requests": self._peak_concurrent_requests,
                "max_concurrent_limit": self._config.max_concurrent_requests,
                "rate_limit_stats": rate_limit_stats,
                # Latency percentiles from tracker (NFR-001 monitoring)
                "p50_latency_ms": latency_stats["p50_latency_ms"],
                "p95_latency_ms": latency_stats["p95_latency_ms"],
                "p99_latency_ms": latency_stats["p99_latency_ms"],
                "min_latency_ms": latency_stats["min_latency_ms"],
                "max_latency_ms": latency_stats["max_latency_ms"],
                "slow_query_count": latency_stats["slow_query_count"],
                "slow_query_threshold_ms": latency_stats["slow_query_threshold_ms"],
            }

    def reset_metrics(self) -> None:
        """Reset metrics counters.

        Thread-safe.
        """
        with self._lock:
            self._total_queries = 0
            self._total_latency_ms = 0.0
            self._rate_limit_events = 0
            self._peak_concurrent_requests = 0
            self._rate_limit_handler.clear_events()
        self._latency_tracker.reset()

    @property
    def rate_limit_handler(self) -> RateLimitHandler:
        """Get the rate limit handler for configuration or monitoring."""
        return self._rate_limit_handler

    @property
    def fallback(self) -> FallbackBehavior:
        """Get the fallback behavior manager for configuration."""
        return self._fallback

    @property
    def concurrent_requests(self) -> int:
        """Get the current number of concurrent requests in flight."""
        with self._lock:
            return self._concurrent_requests

    @property
    def max_concurrent_requests(self) -> int:
        """Get the configured maximum concurrent requests."""
        return self._config.max_concurrent_requests

    @property
    def latency_tracker(self) -> LatencyTracker:
        """Get the latency tracker for monitoring and alerts."""
        return self._latency_tracker

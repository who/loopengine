"""AI behavior engine orchestrator.

This module provides the main AIBehaviorEngine class that coordinates
domain context, prompt building, LLM querying, and response parsing
into a cohesive behavior generation flow.
"""

import logging
import threading
import time
from typing import Any

from loopengine.behaviors.config import LLMConfig, LLMProvider, get_llm_config
from loopengine.behaviors.llm_client import BehaviorResponse, LLMClient, LLMQuery
from loopengine.behaviors.prompt_builder import AgentContext, DomainContext, PromptBuilder
from loopengine.behaviors.providers.claude import ClaudeClient
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
    shared resources.

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
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: ResponseParser | None = None,
        config: LLMConfig | None = None,
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
        """
        self._config = config or get_llm_config()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._response_parser = response_parser or ResponseParser()
        self._llm_client = llm_client or self._create_llm_client()

        # Lock for thread-safe access to shared state
        self._lock = threading.Lock()

        # Metrics tracking
        self._total_queries = 0
        self._total_latency_ms = 0.0

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

    def generate_behavior(
        self,
        domain: DomainContext,
        agent: AgentContext,
        context: dict[str, Any] | None = None,
    ) -> BehaviorResponse:
        """Generate a behavior decision for an agent.

        This is the main entry point for behavior generation. It coordinates:
        1. Building the prompt from domain, agent, and context
        2. Querying the LLM for a behavior decision
        3. Parsing and validating the response

        Thread-safe for concurrent calls.

        Args:
            domain: Domain configuration with type and description.
            agent: Agent information with type and role.
            context: Optional state context dict (current state, nearby agents, etc.)

        Returns:
            BehaviorResponse with action, parameters, reasoning, and metadata
            including latency_ms and provider.

        Raises:
            AIBehaviorEngineError: If behavior generation fails fatally.
        """
        start_time = time.perf_counter()

        try:
            # Build the prompt
            system_message, prompt = self._prompt_builder.build_full_prompt(domain, agent, context)

            # Create the query
            query = LLMQuery(
                prompt=prompt,
                system_message=system_message,
                context=context or {},
            )

            # Query the LLM
            logger.debug(
                "Querying LLM for agent %s in domain %s",
                agent.agent_type,
                domain.domain_type,
            )
            response = self._llm_client.query(query)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics (thread-safe)
            with self._lock:
                self._total_queries += 1
                self._total_latency_ms += latency_ms

            # Enrich metadata with latency and provider info
            enriched_metadata = dict(response.metadata)
            enriched_metadata["latency_ms"] = round(latency_ms, 2)
            enriched_metadata["provider"] = self._config.llm_provider.value

            logger.info(
                "Generated behavior for %s: action=%s, latency=%.2fms",
                agent.agent_type,
                response.action,
                latency_ms,
            )

            return BehaviorResponse(
                action=response.action,
                parameters=response.parameters,
                reasoning=response.reasoning,
                metadata=enriched_metadata,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Behavior generation failed for %s after %.2fms: %s",
                agent.agent_type,
                latency_ms,
                str(e),
            )
            raise AIBehaviorEngineError(f"Failed to generate behavior: {e}") from e

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
            Dict with total_queries, total_latency_ms, and avg_latency_ms.
        """
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._total_queries if self._total_queries > 0 else 0.0
            )
            return {
                "total_queries": self._total_queries,
                "total_latency_ms": round(self._total_latency_ms, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "provider": self._config.llm_provider.value,
            }

    def reset_metrics(self) -> None:
        """Reset metrics counters.

        Thread-safe.
        """
        with self._lock:
            self._total_queries = 0
            self._total_latency_ms = 0.0

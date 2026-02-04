"""Behavior history storage for export and inspection.

This module provides a thread-safe storage mechanism for generated behaviors,
enabling viewing and exporting for debugging, inspection, and documentation.
"""

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoredBehavior:
    """A stored behavior record with full metadata.

    Attributes:
        timestamp: When the behavior was generated (epoch seconds).
        domain_id: ID of the domain configuration used.
        domain_type: Type of domain (e.g., "sandwich shop").
        agent_type: Type of agent that requested the behavior.
        agent_role: Role description of the agent.
        action: The action the agent should take.
        parameters: Parameters for the action.
        reasoning: Explanation of why this action was chosen.
        context: Context provided with the request.
        latency_ms: Time taken to generate the behavior.
        provider: LLM provider used.
        cached: Whether the response was cached.
        fallback: Whether this was a fallback response.
    """

    timestamp: float
    domain_id: str
    domain_type: str
    agent_type: str
    agent_role: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    context: dict[str, Any] | None = None
    latency_ms: float = 0.0
    provider: str = ""
    cached: bool = False
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the stored behavior.
        """
        return {
            "timestamp": self.timestamp,
            "domain_id": self.domain_id,
            "domain_type": self.domain_type,
            "agent_type": self.agent_type,
            "agent_role": self.agent_role,
            "action": self.action,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
            "context": self.context,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "cached": self.cached,
            "fallback": self.fallback,
        }

    def to_human_readable(self) -> str:
        """Convert to human-readable format for documentation.

        Returns:
            Human-readable string representation.
        """
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        lines = [
            f"Behavior Generated: {time_str}",
            f"Domain: {self.domain_type} ({self.domain_id})",
            f"Agent: {self.agent_type} - {self.agent_role}",
            f"Action: {self.action}",
        ]

        if self.parameters:
            lines.append(f"Parameters: {self.parameters}")

        if self.reasoning:
            lines.append(f"Reasoning: {self.reasoning}")

        if self.context:
            lines.append(f"Context: {self.context}")

        lines.append(f"Latency: {self.latency_ms:.2f}ms (Provider: {self.provider})")

        flags = []
        if self.cached:
            flags.append("cached")
        if self.fallback:
            flags.append("fallback")
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")

        return "\n".join(lines)


class BehaviorHistoryStore:
    """Thread-safe storage for generated behaviors.

    Stores behaviors in a fixed-size circular buffer for memory efficiency.
    Supports filtering and export for debugging and documentation.

    Example:
        >>> store = BehaviorHistoryStore(max_size=1000)
        >>> store.record(StoredBehavior(...))
        >>> behaviors = store.export(domain_id="sandwich_shop", limit=50)
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize the behavior history store.

        Args:
            max_size: Maximum number of behaviors to retain (oldest removed first).
        """
        self._max_size = max_size
        self._lock = threading.Lock()
        self._behaviors: collections.deque[StoredBehavior] = collections.deque(maxlen=max_size)
        self._total_recorded = 0

    def record(self, behavior: StoredBehavior) -> None:
        """Record a generated behavior.

        Thread-safe.

        Args:
            behavior: The behavior to store.
        """
        with self._lock:
            self._behaviors.append(behavior)
            self._total_recorded += 1

    def export(
        self,
        domain_id: str | None = None,
        domain_type: str | None = None,
        agent_type: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[StoredBehavior]:
        """Export behaviors with optional filtering.

        Thread-safe.

        Args:
            domain_id: Filter by domain ID.
            domain_type: Filter by domain type.
            agent_type: Filter by agent type.
            start_time: Filter by minimum timestamp (epoch seconds).
            end_time: Filter by maximum timestamp (epoch seconds).
            limit: Maximum number of behaviors to return.
            offset: Number of matching behaviors to skip (for pagination).

        Returns:
            List of matching behaviors, most recent first.
        """
        with self._lock:
            # Filter behaviors
            filtered: list[StoredBehavior] = []
            for behavior in reversed(self._behaviors):
                # Apply filters
                if domain_id is not None and behavior.domain_id != domain_id:
                    continue
                if domain_type is not None and behavior.domain_type != domain_type:
                    continue
                if agent_type is not None and behavior.agent_type != agent_type:
                    continue
                if start_time is not None and behavior.timestamp < start_time:
                    continue
                if end_time is not None and behavior.timestamp > end_time:
                    continue
                filtered.append(behavior)

            # Apply offset and limit
            return filtered[offset : offset + limit]

    def export_json(
        self,
        domain_id: str | None = None,
        domain_type: str | None = None,
        agent_type: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Export behaviors as JSON-serializable dictionaries.

        Args:
            domain_id: Filter by domain ID.
            domain_type: Filter by domain type.
            agent_type: Filter by agent type.
            start_time: Filter by minimum timestamp (epoch seconds).
            end_time: Filter by maximum timestamp (epoch seconds).
            limit: Maximum number of behaviors to return.
            offset: Number of matching behaviors to skip.

        Returns:
            List of behavior dictionaries, most recent first.
        """
        behaviors = self.export(
            domain_id=domain_id,
            domain_type=domain_type,
            agent_type=agent_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
        return [b.to_dict() for b in behaviors]

    def export_human_readable(
        self,
        domain_id: str | None = None,
        domain_type: str | None = None,
        agent_type: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> str:
        """Export behaviors as human-readable text.

        Args:
            domain_id: Filter by domain ID.
            domain_type: Filter by domain type.
            agent_type: Filter by agent type.
            start_time: Filter by minimum timestamp (epoch seconds).
            end_time: Filter by maximum timestamp (epoch seconds).
            limit: Maximum number of behaviors to return.
            offset: Number of matching behaviors to skip.

        Returns:
            Human-readable string with all matching behaviors.
        """
        behaviors = self.export(
            domain_id=domain_id,
            domain_type=domain_type,
            agent_type=agent_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )

        if not behaviors:
            return "No behaviors found matching the specified filters."

        separator = "\n" + "=" * 60 + "\n"
        return separator.join(b.to_human_readable() for b in behaviors)

    def count(
        self,
        domain_id: str | None = None,
        domain_type: str | None = None,
        agent_type: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> int:
        """Count behaviors matching the filters.

        Thread-safe.

        Args:
            domain_id: Filter by domain ID.
            domain_type: Filter by domain type.
            agent_type: Filter by agent type.
            start_time: Filter by minimum timestamp (epoch seconds).
            end_time: Filter by maximum timestamp (epoch seconds).

        Returns:
            Number of matching behaviors.
        """
        with self._lock:
            count = 0
            for behavior in self._behaviors:
                if domain_id is not None and behavior.domain_id != domain_id:
                    continue
                if domain_type is not None and behavior.domain_type != domain_type:
                    continue
                if agent_type is not None and behavior.agent_type != agent_type:
                    continue
                if start_time is not None and behavior.timestamp < start_time:
                    continue
                if end_time is not None and behavior.timestamp > end_time:
                    continue
                count += 1
            return count

    def get_domains(self) -> list[str]:
        """Get list of unique domain IDs in the history.

        Thread-safe.

        Returns:
            List of unique domain IDs.
        """
        with self._lock:
            return list({b.domain_id for b in self._behaviors})

    def get_agent_types(self, domain_id: str | None = None) -> list[str]:
        """Get list of unique agent types, optionally filtered by domain.

        Thread-safe.

        Args:
            domain_id: Optional domain ID filter.

        Returns:
            List of unique agent types.
        """
        with self._lock:
            if domain_id is None:
                return list({b.agent_type for b in self._behaviors})
            return list({b.agent_type for b in self._behaviors if b.domain_id == domain_id})

    def clear(self) -> None:
        """Clear all stored behaviors.

        Thread-safe.
        """
        with self._lock:
            self._behaviors.clear()

    @property
    def size(self) -> int:
        """Get current number of stored behaviors."""
        with self._lock:
            return len(self._behaviors)

    @property
    def total_recorded(self) -> int:
        """Get total number of behaviors recorded (including removed)."""
        with self._lock:
            return self._total_recorded

    @property
    def max_size(self) -> int:
        """Get maximum storage size."""
        return self._max_size

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the behavior history.

        Thread-safe.

        Returns:
            Dictionary with size, total_recorded, max_size, unique_domains,
            unique_agent_types.
        """
        with self._lock:
            domains = {b.domain_id for b in self._behaviors}
            agent_types = {b.agent_type for b in self._behaviors}
            return {
                "size": len(self._behaviors),
                "total_recorded": self._total_recorded,
                "max_size": self._max_size,
                "unique_domains": len(domains),
                "unique_agent_types": len(agent_types),
            }

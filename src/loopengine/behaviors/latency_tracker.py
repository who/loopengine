"""LLM query latency monitoring and alerting.

This module provides latency tracking for LLM queries to ensure NFR-001 compliance
(95% of requests under 3 seconds). Features include:
- Per-query latency tracking
- Percentile calculations (p50, p95, p99)
- Slow query detection and logging
- Alert mechanism for sustained high latency
"""

import collections
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# NFR-001: 95% of requests under 3 seconds
SLOW_QUERY_THRESHOLD_MS = 3000.0


class AlertSeverity(StrEnum):
    """Severity levels for latency alerts."""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SlowQueryEvent:
    """Record of a slow LLM query.

    Attributes:
        timestamp: When the query occurred (epoch seconds).
        latency_ms: Query latency in milliseconds.
        agent_type: Type of agent that made the query.
        domain_type: Domain type for the query.
        context: Additional context about the query.
    """

    timestamp: float
    latency_ms: float
    agent_type: str
    domain_type: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyAlert:
    """Alert for sustained high latency.

    Attributes:
        timestamp: When the alert was triggered.
        severity: Alert severity level.
        message: Human-readable alert message.
        p95_latency_ms: Current p95 latency when alert triggered.
        window_size: Number of queries in the evaluation window.
        slow_query_count: Number of slow queries in the window.
    """

    timestamp: float
    severity: AlertSeverity
    message: str
    p95_latency_ms: float
    window_size: int
    slow_query_count: int


class LatencyTracker:
    """Thread-safe latency tracker for LLM queries.

    Tracks individual query latencies, calculates percentiles, logs slow queries,
    and triggers alerts for sustained high latency.

    Example:
        >>> tracker = LatencyTracker(max_history=1000)
        >>> tracker.record(150.5, "cashier", "sandwich_shop", {"order_id": 123})
        >>> stats = tracker.get_stats()
        >>> print(f"p95: {stats['p95_latency_ms']}ms")
    """

    def __init__(
        self,
        max_history: int = 1000,
        slow_threshold_ms: float = SLOW_QUERY_THRESHOLD_MS,
        alert_callback: Callable[[LatencyAlert], None] | None = None,
        alert_window_size: int = 100,
        alert_threshold_pct: float = 10.0,
    ) -> None:
        """Initialize the latency tracker.

        Args:
            max_history: Maximum number of latency records to retain.
            slow_threshold_ms: Threshold in ms for slow query logging.
            alert_callback: Callback function for latency alerts.
            alert_window_size: Number of recent queries to evaluate for alerts.
            alert_threshold_pct: Percentage of slow queries that triggers an alert.
        """
        self._max_history = max_history
        self._slow_threshold_ms = slow_threshold_ms
        self._alert_callback = alert_callback
        self._alert_window_size = alert_window_size
        self._alert_threshold_pct = alert_threshold_pct

        self._lock = threading.Lock()
        self._latencies: collections.deque[float] = collections.deque(maxlen=max_history)
        self._slow_queries: collections.deque[SlowQueryEvent] = collections.deque(
            maxlen=max_history
        )
        self._alerts: collections.deque[LatencyAlert] = collections.deque(maxlen=100)
        self._total_queries = 0
        self._total_latency_ms = 0.0

        # Alert cooldown to prevent spam (minimum 60 seconds between alerts)
        self._last_alert_time = 0.0
        self._alert_cooldown_seconds = 60.0

    def record(
        self,
        latency_ms: float,
        agent_type: str = "",
        domain_type: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a query latency measurement.

        Thread-safe. Logs slow queries and checks for alert conditions.

        Args:
            latency_ms: Query latency in milliseconds.
            agent_type: Type of agent that made the query.
            domain_type: Domain type for the query.
            context: Additional context about the query.
        """
        now = time.time()
        context = context or {}

        with self._lock:
            self._latencies.append(latency_ms)
            self._total_queries += 1
            self._total_latency_ms += latency_ms

            # Check for slow query
            if latency_ms > self._slow_threshold_ms:
                event = SlowQueryEvent(
                    timestamp=now,
                    latency_ms=latency_ms,
                    agent_type=agent_type,
                    domain_type=domain_type,
                    context=context,
                )
                self._slow_queries.append(event)
                logger.warning(
                    "Slow LLM query detected: %.2fms (threshold: %.0fms) "
                    "agent_type=%s domain=%s context=%s",
                    latency_ms,
                    self._slow_threshold_ms,
                    agent_type,
                    domain_type,
                    context,
                )

            # Check alert conditions after recording
            self._check_alert_conditions()

    def _check_alert_conditions(self) -> None:
        """Check if alert conditions are met.

        Must be called with lock held.
        """
        now = time.time()

        # Respect cooldown period
        if now - self._last_alert_time < self._alert_cooldown_seconds:
            return

        # Need enough data to evaluate
        if len(self._latencies) < self._alert_window_size:
            return

        # Get recent latencies for evaluation
        recent = list(self._latencies)[-self._alert_window_size :]
        slow_count = sum(1 for lat in recent if lat > self._slow_threshold_ms)
        slow_pct = (slow_count / len(recent)) * 100

        # Check if p95 is above threshold (NFR-001 violation)
        sorted_recent = sorted(recent)
        p95_idx = int(len(sorted_recent) * 0.95)
        p95_latency = sorted_recent[p95_idx] if sorted_recent else 0.0

        if slow_pct >= self._alert_threshold_pct or p95_latency > self._slow_threshold_ms:
            severity = AlertSeverity.CRITICAL if slow_pct >= 20 else AlertSeverity.WARNING

            alert = LatencyAlert(
                timestamp=now,
                severity=severity,
                message=(
                    f"Sustained high latency detected: {slow_pct:.1f}% of queries "
                    f"over {self._slow_threshold_ms}ms in last {len(recent)} queries. "
                    f"p95={p95_latency:.2f}ms"
                ),
                p95_latency_ms=p95_latency,
                window_size=len(recent),
                slow_query_count=slow_count,
            )

            self._alerts.append(alert)
            self._last_alert_time = now

            logger.warning("Latency alert triggered: %s", alert.message)

            # Call callback outside lock if registered
            if self._alert_callback:
                # Release lock temporarily for callback
                callback = self._alert_callback
                # Note: we're still holding the lock here, but the callback
                # shouldn't need to access tracker internals
                try:
                    callback(alert)
                except Exception as e:
                    logger.error("Alert callback error: %s", e)

    def get_stats(self) -> dict[str, Any]:
        """Get current latency statistics.

        Thread-safe.

        Returns:
            Dictionary with:
            - total_queries: Total number of queries tracked
            - avg_latency_ms: Average latency in milliseconds
            - p50_latency_ms: 50th percentile latency
            - p95_latency_ms: 95th percentile latency
            - p99_latency_ms: 99th percentile latency
            - min_latency_ms: Minimum latency observed
            - max_latency_ms: Maximum latency observed
            - slow_query_count: Number of slow queries
            - slow_query_threshold_ms: Threshold for slow queries
            - history_size: Current number of latencies in history
        """
        with self._lock:
            if not self._latencies:
                return {
                    "total_queries": self._total_queries,
                    "avg_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "min_latency_ms": 0.0,
                    "max_latency_ms": 0.0,
                    "slow_query_count": len(self._slow_queries),
                    "slow_query_threshold_ms": self._slow_threshold_ms,
                    "history_size": 0,
                }

            sorted_latencies = sorted(self._latencies)
            n = len(sorted_latencies)

            def percentile(p: float) -> float:
                """Calculate percentile from sorted list."""
                if n == 0:
                    return 0.0
                idx = int(n * p / 100)
                idx = min(idx, n - 1)
                return sorted_latencies[idx]

            avg_latency = self._total_latency_ms / self._total_queries

            return {
                "total_queries": self._total_queries,
                "avg_latency_ms": round(avg_latency, 2),
                "p50_latency_ms": round(percentile(50), 2),
                "p95_latency_ms": round(percentile(95), 2),
                "p99_latency_ms": round(percentile(99), 2),
                "min_latency_ms": round(min(sorted_latencies), 2),
                "max_latency_ms": round(max(sorted_latencies), 2),
                "slow_query_count": len(self._slow_queries),
                "slow_query_threshold_ms": self._slow_threshold_ms,
                "history_size": n,
            }

    def get_slow_queries(self, limit: int = 100) -> list[SlowQueryEvent]:
        """Get recent slow query events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of slow query events, most recent first.
        """
        with self._lock:
            queries = list(self._slow_queries)
            queries.reverse()  # Most recent first
            return queries[:limit]

    def get_alerts(self, limit: int = 100) -> list[LatencyAlert]:
        """Get recent latency alerts.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of alerts, most recent first.
        """
        with self._lock:
            alerts = list(self._alerts)
            alerts.reverse()  # Most recent first
            return alerts[:limit]

    def set_alert_callback(self, callback: Callable[[LatencyAlert], None] | None) -> None:
        """Set or clear the alert callback.

        Args:
            callback: Callback function or None to clear.
        """
        with self._lock:
            self._alert_callback = callback

    def reset(self) -> None:
        """Reset all tracked data.

        Thread-safe.
        """
        with self._lock:
            self._latencies.clear()
            self._slow_queries.clear()
            self._alerts.clear()
            self._total_queries = 0
            self._total_latency_ms = 0.0
            self._last_alert_time = 0.0

    @property
    def slow_threshold_ms(self) -> float:
        """Get the slow query threshold in milliseconds."""
        return self._slow_threshold_ms

    @slow_threshold_ms.setter
    def slow_threshold_ms(self, value: float) -> None:
        """Set the slow query threshold in milliseconds."""
        with self._lock:
            self._slow_threshold_ms = value

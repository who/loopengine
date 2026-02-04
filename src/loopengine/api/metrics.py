"""API endpoints for LLM metrics and monitoring.

This module provides REST API endpoints for monitoring LLM query latencies
to ensure NFR-001 compliance (95% of requests under 3 seconds).
"""

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from loopengine.behaviors import AIBehaviorEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


class LatencyMetrics(BaseModel):
    """Latency statistics for LLM queries.

    Attributes:
        total_queries: Total number of queries tracked.
        avg_latency_ms: Average latency in milliseconds.
        p50_latency_ms: 50th percentile latency.
        p95_latency_ms: 95th percentile latency.
        p99_latency_ms: 99th percentile latency.
        min_latency_ms: Minimum latency observed.
        max_latency_ms: Maximum latency observed.
        slow_query_count: Number of slow queries (>3 seconds).
        slow_query_threshold_ms: Threshold for slow queries.
        history_size: Current number of latencies in history.
    """

    total_queries: int = Field(description="Total number of queries tracked")
    avg_latency_ms: float = Field(description="Average latency in milliseconds")
    p50_latency_ms: float = Field(description="50th percentile latency")
    p95_latency_ms: float = Field(description="95th percentile latency")
    p99_latency_ms: float = Field(description="99th percentile latency")
    min_latency_ms: float = Field(description="Minimum latency observed")
    max_latency_ms: float = Field(description="Maximum latency observed")
    slow_query_count: int = Field(description="Number of slow queries (>3 seconds)")
    slow_query_threshold_ms: float = Field(description="Threshold for slow queries")
    history_size: int = Field(description="Number of latencies in history")


class ConcurrencyMetrics(BaseModel):
    """Concurrency metrics for the behavior engine.

    Attributes:
        concurrent_requests: Current number of concurrent requests.
        peak_concurrent_requests: Peak concurrent requests observed.
        max_concurrent_limit: Maximum allowed concurrent requests.
    """

    concurrent_requests: int = Field(description="Current concurrent requests")
    peak_concurrent_requests: int = Field(description="Peak concurrent requests")
    max_concurrent_limit: int = Field(description="Maximum allowed concurrent requests")


class RateLimitMetrics(BaseModel):
    """Rate limit metrics.

    Attributes:
        rate_limit_events: Number of rate limit events.
        total_retries: Total retries due to rate limits.
        fallbacks_used: Number of times fallback was used.
    """

    rate_limit_events: int = Field(description="Number of rate limit events")
    total_retries: int = Field(default=0, description="Total retries due to rate limits")
    fallbacks_used: int = Field(default=0, description="Number of fallbacks used")


class SlowQueryResponse(BaseModel):
    """A slow query event.

    Attributes:
        timestamp: When the query occurred (epoch seconds).
        latency_ms: Query latency in milliseconds.
        agent_type: Type of agent that made the query.
        domain_type: Domain type for the query.
        context: Additional context about the query.
    """

    timestamp: float = Field(description="When the query occurred")
    latency_ms: float = Field(description="Query latency in milliseconds")
    agent_type: str = Field(description="Agent type")
    domain_type: str = Field(description="Domain type")
    context: dict[str, Any] = Field(default_factory=dict, description="Query context")


class AlertResponse(BaseModel):
    """A latency alert.

    Attributes:
        timestamp: When the alert was triggered.
        severity: Alert severity (warning or critical).
        message: Human-readable alert message.
        p95_latency_ms: p95 latency when alert triggered.
        window_size: Number of queries in evaluation window.
        slow_query_count: Number of slow queries in window.
    """

    timestamp: float = Field(description="When the alert was triggered")
    severity: str = Field(description="Alert severity (warning/critical)")
    message: str = Field(description="Human-readable alert message")
    p95_latency_ms: float = Field(description="p95 latency when alert triggered")
    window_size: int = Field(description="Queries in evaluation window")
    slow_query_count: int = Field(description="Slow queries in window")


class MetricsResponse(BaseModel):
    """Complete metrics response.

    Attributes:
        provider: LLM provider being used.
        latency: Latency statistics.
        concurrency: Concurrency metrics.
        rate_limits: Rate limit metrics.
    """

    provider: str = Field(description="LLM provider being used")
    latency: LatencyMetrics = Field(description="Latency statistics")
    concurrency: ConcurrencyMetrics = Field(description="Concurrency metrics")
    rate_limits: RateLimitMetrics = Field(description="Rate limit metrics")


# Engine instance cache for reuse (shared with behaviors API)
_engine: AIBehaviorEngine | None = None


def _get_engine() -> AIBehaviorEngine:
    """Get or create an AIBehaviorEngine instance.

    Returns:
        AIBehaviorEngine instance.
    """
    global _engine
    if _engine is None:
        _engine = AIBehaviorEngine()
    return _engine


def set_engine(engine: AIBehaviorEngine) -> None:
    """Set the engine instance for testing or shared use.

    Args:
        engine: AIBehaviorEngine instance to use.
    """
    global _engine
    _engine = engine


@router.get(
    "",
    response_model=MetricsResponse,
    responses={
        200: {"description": "Metrics retrieved successfully"},
    },
)
async def get_metrics() -> MetricsResponse:
    """Get current LLM query metrics.

    Returns metrics including:
    - Latency percentiles (p50, p95, p99)
    - Slow query counts
    - Concurrency statistics
    - Rate limit events

    This endpoint supports NFR-001 monitoring (95% of requests under 3 seconds).

    Returns:
        MetricsResponse with current metrics.
    """
    engine = _get_engine()
    metrics = engine.metrics
    latency_stats = engine.latency_tracker.get_stats()

    return MetricsResponse(
        provider=metrics["provider"],
        latency=LatencyMetrics(
            total_queries=latency_stats["total_queries"],
            avg_latency_ms=latency_stats["avg_latency_ms"],
            p50_latency_ms=latency_stats["p50_latency_ms"],
            p95_latency_ms=latency_stats["p95_latency_ms"],
            p99_latency_ms=latency_stats["p99_latency_ms"],
            min_latency_ms=latency_stats["min_latency_ms"],
            max_latency_ms=latency_stats["max_latency_ms"],
            slow_query_count=latency_stats["slow_query_count"],
            slow_query_threshold_ms=latency_stats["slow_query_threshold_ms"],
            history_size=latency_stats["history_size"],
        ),
        concurrency=ConcurrencyMetrics(
            concurrent_requests=metrics["concurrent_requests"],
            peak_concurrent_requests=metrics["peak_concurrent_requests"],
            max_concurrent_limit=metrics["max_concurrent_limit"],
        ),
        rate_limits=RateLimitMetrics(
            rate_limit_events=metrics["rate_limit_events"],
            total_retries=metrics.get("rate_limit_stats", {}).get("total_retries", 0),
            fallbacks_used=metrics.get("rate_limit_stats", {}).get("fallbacks_used", 0),
        ),
    )


@router.get(
    "/slow-queries",
    response_model=list[SlowQueryResponse],
    responses={
        200: {"description": "Slow queries retrieved successfully"},
    },
)
async def get_slow_queries(limit: int = 100) -> list[SlowQueryResponse]:
    """Get recent slow query events.

    Slow queries are those exceeding the threshold (default 3 seconds).
    Logs include context for debugging.

    Args:
        limit: Maximum number of events to return (default 100).

    Returns:
        List of slow query events, most recent first.
    """
    engine = _get_engine()
    events = engine.latency_tracker.get_slow_queries(limit=limit)

    return [
        SlowQueryResponse(
            timestamp=event.timestamp,
            latency_ms=event.latency_ms,
            agent_type=event.agent_type,
            domain_type=event.domain_type,
            context=event.context,
        )
        for event in events
    ]


@router.get(
    "/alerts",
    response_model=list[AlertResponse],
    responses={
        200: {"description": "Alerts retrieved successfully"},
    },
)
async def get_alerts(limit: int = 100) -> list[AlertResponse]:
    """Get recent latency alerts.

    Alerts are triggered when sustained high latency is detected
    (e.g., >10% of queries over 3 seconds in the last 100 queries).

    Args:
        limit: Maximum number of alerts to return (default 100).

    Returns:
        List of alerts, most recent first.
    """
    engine = _get_engine()
    alerts = engine.latency_tracker.get_alerts(limit=limit)

    return [
        AlertResponse(
            timestamp=alert.timestamp,
            severity=alert.severity.value,
            message=alert.message,
            p95_latency_ms=alert.p95_latency_ms,
            window_size=alert.window_size,
            slow_query_count=alert.slow_query_count,
        )
        for alert in alerts
    ]


@router.post(
    "/reset",
    responses={
        200: {"description": "Metrics reset successfully"},
    },
)
async def reset_metrics() -> dict[str, str]:
    """Reset all metrics counters.

    Clears:
    - Latency history and percentiles
    - Slow query events
    - Alerts
    - Rate limit counters
    - Concurrency peak values

    Returns:
        Success message.
    """
    engine = _get_engine()
    engine.reset_metrics()
    logger.info("Metrics reset via API")
    return {"message": "Metrics reset successfully"}

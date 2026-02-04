"""API endpoints for behavior generation.

This module provides REST API endpoints for generating agent behaviors
using the AIBehaviorEngine and domain configurations, as well as viewing
and exporting generated behaviors for inspection and documentation.
"""

import logging
import time
from enum import StrEnum
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from loopengine.behaviors import (
    AIBehaviorEngine,
    AIBehaviorEngineError,
    BehaviorCache,
    BehaviorHistoryStore,
    DomainStore,
    DomainStoreError,
    StoredBehavior,
)
from loopengine.behaviors.prompt_builder import AgentContext, ConstraintContext, DomainContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/behaviors", tags=["behaviors"])


class GenerateBehaviorRequest(BaseModel):
    """Request body for generating a behavior.

    Attributes:
        domain_id: ID of the domain configuration to use.
        agent_type: Type of agent requesting behavior decision.
        context: Optional context dictionary with current state.
    """

    domain_id: str = Field(
        description="ID of the domain configuration",
        min_length=1,
    )
    agent_type: str = Field(
        description="Type of agent requesting behavior",
        min_length=1,
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="Current context/state for the agent",
    )

    @field_validator("domain_id")
    @classmethod
    def domain_id_not_empty(cls, v: str) -> str:
        """Validate domain_id is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("domain_id cannot be empty or whitespace only")
        return v.strip()

    @field_validator("agent_type")
    @classmethod
    def agent_type_not_empty(cls, v: str) -> str:
        """Validate agent_type is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("agent_type cannot be empty or whitespace only")
        return v.strip()


class BehaviorMetadata(BaseModel):
    """Metadata about the behavior generation.

    Attributes:
        cached: Whether the response was served from cache.
        latency_ms: Time taken to generate the behavior in milliseconds.
        provider: The LLM provider used for generation.
    """

    cached: bool = Field(default=False, description="Whether response was cached")
    latency_ms: float = Field(description="Generation latency in milliseconds")
    provider: str = Field(description="LLM provider used")


class GenerateBehaviorResponse(BaseModel):
    """Response body for behavior generation.

    Attributes:
        action: The action the agent should take.
        parameters: Parameters for the action.
        reasoning: Brief explanation of why this action was chosen.
        metadata: Generation metadata including latency and provider.
    """

    action: str = Field(description="The action the agent should take")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the action",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why this action was chosen",
    )
    metadata: BehaviorMetadata = Field(description="Generation metadata")


# Engine instance cache for reuse
_engine: AIBehaviorEngine | None = None
_history_store: BehaviorHistoryStore | None = None
_behavior_cache: BehaviorCache | None = None


def _get_engine() -> AIBehaviorEngine:
    """Get or create an AIBehaviorEngine instance.

    Returns:
        AIBehaviorEngine instance.
    """
    global _engine
    if _engine is None:
        _engine = AIBehaviorEngine()
    return _engine


def _get_store() -> DomainStore:
    """Get or create a DomainStore instance.

    Returns:
        DomainStore instance.
    """
    return DomainStore()


def _get_history_store() -> BehaviorHistoryStore:
    """Get or create a BehaviorHistoryStore instance.

    Returns:
        BehaviorHistoryStore instance.
    """
    global _history_store
    if _history_store is None:
        _history_store = BehaviorHistoryStore(max_size=10000)
    return _history_store


def set_history_store(store: BehaviorHistoryStore) -> None:
    """Set the history store instance (for testing).

    Args:
        store: BehaviorHistoryStore instance to use.
    """
    global _history_store
    _history_store = store


def _get_behavior_cache() -> BehaviorCache:
    """Get or create a BehaviorCache instance.

    Returns:
        BehaviorCache instance.
    """
    global _behavior_cache
    if _behavior_cache is None:
        _behavior_cache = BehaviorCache()
    return _behavior_cache


def set_behavior_cache(cache: BehaviorCache) -> None:
    """Set the behavior cache instance (for testing).

    Args:
        cache: BehaviorCache instance to use.
    """
    global _behavior_cache
    _behavior_cache = cache


def _find_agent_type_in_schema(
    agent_type: str,
    agent_types: list[Any],
) -> tuple[str, str]:
    """Find an agent type in the schema and return its role.

    Args:
        agent_type: The agent type to find.
        agent_types: List of agent type schemas from the domain.

    Returns:
        Tuple of (normalized_agent_type, agent_role).

    Raises:
        ValueError: If agent type is not found in schema.
    """
    # Normalize the input agent type for comparison
    normalized_input = agent_type.lower().replace("-", "_").replace(" ", "_")

    for at in agent_types:
        # Handle both dict and object access
        if hasattr(at, "name"):
            name = at.name
            role = at.role
        else:
            name = at.get("name", "")
            role = at.get("role", "")

        normalized_name = name.lower().replace("-", "_").replace(" ", "_")
        if normalized_name == normalized_input:
            return name, role

    # Return list of valid agent types for error message
    valid_types = []
    for at in agent_types:
        if hasattr(at, "name"):
            valid_types.append(at.name)
        else:
            valid_types.append(at.get("name", "unknown"))

    raise ValueError(
        f"Agent type '{agent_type}' not found in domain. Valid types: {', '.join(valid_types)}"
    )


@router.post(
    "/generate",
    response_model=GenerateBehaviorResponse,
    responses={
        200: {"description": "Behavior generated successfully"},
        400: {"description": "Invalid request - invalid agent_type for domain"},
        404: {"description": "Domain not found"},
        500: {"description": "Internal error - LLM or generation failed"},
        503: {"description": "Service unavailable - LLM provider error"},
    },
)
async def generate_behavior(
    request: GenerateBehaviorRequest,
) -> GenerateBehaviorResponse:
    """Generate a behavior decision for an agent given domain and context.

    Uses the AIBehaviorEngine to query the configured LLM provider and generate
    an appropriate action for the agent based on the domain configuration and
    current context.

    Args:
        request: Request containing domain_id, agent_type, and optional context.

    Returns:
        GenerateBehaviorResponse with action, parameters, reasoning, and metadata.

    Raises:
        HTTPException: 404 if domain not found, 400 if agent_type invalid,
            500/503 for LLM errors.
    """
    start_time = time.perf_counter()
    store = _get_store()
    engine = _get_engine()

    # Load domain configuration
    try:
        stored = store.load(request.domain_id)
    except DomainStoreError as e:
        logger.warning("Domain not found: %s", request.domain_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{request.domain_id}' not found",
        ) from e

    # Validate agent_type exists in domain
    try:
        agent_name, agent_role = _find_agent_type_in_schema(
            request.agent_type,
            stored.schema_.agent_types,
        )
    except ValueError as e:
        logger.warning("Invalid agent type for domain %s: %s", request.domain_id, str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Build contexts for the engine
    # Convert domain constraints to context constraints
    constraints = [
        ConstraintContext(text=c.text, constraint_type=c.constraint_type)
        for c in stored.schema_.constraints
    ]
    domain_context = DomainContext(
        domain_type=stored.schema_.domain_type,
        domain_description=stored.schema_.description,
        constraints=constraints,
    )
    agent_context = AgentContext(
        agent_type=agent_name,
        agent_role=agent_role,
    )

    # Generate behavior
    try:
        behavior_response = engine.generate_behavior(
            domain=domain_context,
            agent=agent_context,
            context=request.context,
        )
    except AIBehaviorEngineError as e:
        error_msg = str(e).lower()
        # Check for rate limit or provider errors
        if "rate limit" in error_msg or "quota" in error_msg:
            logger.error("LLM rate limit hit: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM provider rate limit exceeded. Please retry later.",
            ) from e
        if "api" in error_msg and ("key" in error_msg or "auth" in error_msg):
            logger.error("LLM authentication error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM provider authentication error. Please check configuration.",
            ) from e
        # Generic LLM error
        logger.error("Behavior generation failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate behavior. Please try again.",
        ) from e
    except Exception as e:
        logger.error("Unexpected error during behavior generation: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during behavior generation",
        ) from e

    # Calculate total latency including domain lookup
    total_latency_ms = (time.perf_counter() - start_time) * 1000

    # Build response metadata
    # Use latency from the behavior engine if available, otherwise use total
    engine_latency = behavior_response.metadata.get("latency_ms", total_latency_ms)
    provider = behavior_response.metadata.get("provider", engine.provider)
    cached = behavior_response.metadata.get("cached", False)

    logger.info(
        "Generated behavior for %s/%s: action=%s, latency=%.2fms",
        request.domain_id,
        agent_name,
        behavior_response.action,
        total_latency_ms,
    )

    # Record behavior to history for export
    history_store = _get_history_store()
    stored_behavior = StoredBehavior(
        timestamp=time.time(),
        domain_id=request.domain_id,
        domain_type=stored.schema_.domain_type,
        agent_type=agent_name,
        agent_role=agent_role,
        action=behavior_response.action,
        parameters=behavior_response.parameters,
        reasoning=behavior_response.reasoning,
        context=request.context,
        latency_ms=round(engine_latency, 2),
        provider=provider,
        cached=cached,
        fallback=behavior_response.metadata.get("fallback", False),
    )
    history_store.record(stored_behavior)

    return GenerateBehaviorResponse(
        action=behavior_response.action,
        parameters=behavior_response.parameters,
        reasoning=behavior_response.reasoning,
        metadata=BehaviorMetadata(
            cached=cached,
            latency_ms=round(engine_latency, 2),
            provider=provider,
        ),
    )


class ExportFormat(StrEnum):
    """Export format options."""

    JSON = "json"
    HUMAN_READABLE = "human_readable"


class ExportedBehavior(BaseModel):
    """Exported behavior record.

    Attributes:
        timestamp: When the behavior was generated (epoch seconds).
        domain_id: ID of the domain configuration used.
        domain_type: Type of domain.
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

    timestamp: float = Field(description="When the behavior was generated (epoch)")
    domain_id: str = Field(description="Domain configuration ID")
    domain_type: str = Field(description="Type of domain")
    agent_type: str = Field(description="Agent type")
    agent_role: str = Field(description="Agent role description")
    action: str = Field(description="The action chosen")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    reasoning: str = Field(default="", description="Reasoning for the action")
    context: dict[str, Any] | None = Field(default=None, description="Context provided")
    latency_ms: float = Field(description="Generation latency in milliseconds")
    provider: str = Field(description="LLM provider used")
    cached: bool = Field(description="Whether response was cached")
    fallback: bool = Field(description="Whether this was a fallback response")


class ExportResponse(BaseModel):
    """Response for behavior export.

    Attributes:
        behaviors: List of exported behaviors.
        total_count: Total number of matching behaviors.
        offset: Pagination offset used.
        limit: Pagination limit used.
    """

    behaviors: list[ExportedBehavior] = Field(description="Exported behaviors")
    total_count: int = Field(description="Total matching behaviors")
    offset: int = Field(description="Pagination offset")
    limit: int = Field(description="Pagination limit")


class ExportStatsResponse(BaseModel):
    """Response for behavior export statistics.

    Attributes:
        size: Current number of stored behaviors.
        total_recorded: Total behaviors recorded since startup.
        max_size: Maximum storage capacity.
        unique_domains: Number of unique domains.
        unique_agent_types: Number of unique agent types.
    """

    size: int = Field(description="Current number of stored behaviors")
    total_recorded: int = Field(description="Total behaviors ever recorded")
    max_size: int = Field(description="Maximum storage capacity")
    unique_domains: int = Field(description="Number of unique domains")
    unique_agent_types: int = Field(description="Number of unique agent types")


@router.get(
    "/export",
    response_model=ExportResponse,
    responses={
        200: {"description": "Behaviors exported successfully"},
    },
)
async def export_behaviors(
    domain_id: str | None = Query(default=None, description="Filter by domain ID"),
    domain_type: str | None = Query(default=None, description="Filter by domain type"),
    agent_type: str | None = Query(default=None, description="Filter by agent type"),
    start_time: float | None = Query(
        default=None, description="Filter by minimum timestamp (epoch seconds)"
    ),
    end_time: float | None = Query(
        default=None, description="Filter by maximum timestamp (epoch seconds)"
    ),
    limit: int = Query(default=100, ge=1, le=10000, description="Maximum behaviors to return"),
    offset: int = Query(default=0, ge=0, description="Number of behaviors to skip"),
) -> ExportResponse:
    """Export generated behaviors with optional filtering.

    Supports filtering by domain, agent type, and time range. Results are
    returned in reverse chronological order (most recent first).

    Use for inspection, debugging, and documentation of agent behaviors.

    Args:
        domain_id: Optional filter by domain ID.
        domain_type: Optional filter by domain type.
        agent_type: Optional filter by agent type.
        start_time: Optional minimum timestamp (epoch seconds).
        end_time: Optional maximum timestamp (epoch seconds).
        limit: Maximum number of behaviors to return (1-10000).
        offset: Number of matching behaviors to skip for pagination.

    Returns:
        ExportResponse with matching behaviors and pagination info.
    """
    history_store = _get_history_store()

    # Get total count for pagination info
    total_count = history_store.count(
        domain_id=domain_id,
        domain_type=domain_type,
        agent_type=agent_type,
        start_time=start_time,
        end_time=end_time,
    )

    # Get behaviors
    behaviors = history_store.export(
        domain_id=domain_id,
        domain_type=domain_type,
        agent_type=agent_type,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )

    exported = [
        ExportedBehavior(
            timestamp=b.timestamp,
            domain_id=b.domain_id,
            domain_type=b.domain_type,
            agent_type=b.agent_type,
            agent_role=b.agent_role,
            action=b.action,
            parameters=b.parameters,
            reasoning=b.reasoning,
            context=b.context,
            latency_ms=b.latency_ms,
            provider=b.provider,
            cached=b.cached,
            fallback=b.fallback,
        )
        for b in behaviors
    ]

    return ExportResponse(
        behaviors=exported,
        total_count=total_count,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/export/text",
    responses={
        200: {
            "description": "Behaviors exported as human-readable text",
            "content": {"text/plain": {}},
        },
    },
)
async def export_behaviors_text(
    domain_id: str | None = Query(default=None, description="Filter by domain ID"),
    domain_type: str | None = Query(default=None, description="Filter by domain type"),
    agent_type: str | None = Query(default=None, description="Filter by agent type"),
    start_time: float | None = Query(
        default=None, description="Filter by minimum timestamp (epoch seconds)"
    ),
    end_time: float | None = Query(
        default=None, description="Filter by maximum timestamp (epoch seconds)"
    ),
    limit: int = Query(default=100, ge=1, le=10000, description="Maximum behaviors to return"),
    offset: int = Query(default=0, ge=0, description="Number of behaviors to skip"),
) -> str:
    """Export generated behaviors as human-readable text.

    Same filtering options as /export but returns plain text format
    suitable for documentation and reports.

    Args:
        domain_id: Optional filter by domain ID.
        domain_type: Optional filter by domain type.
        agent_type: Optional filter by agent type.
        start_time: Optional minimum timestamp (epoch seconds).
        end_time: Optional maximum timestamp (epoch seconds).
        limit: Maximum number of behaviors to return (1-10000).
        offset: Number of matching behaviors to skip for pagination.

    Returns:
        Human-readable text representation of behaviors.
    """
    from fastapi.responses import PlainTextResponse

    history_store = _get_history_store()

    text = history_store.export_human_readable(
        domain_id=domain_id,
        domain_type=domain_type,
        agent_type=agent_type,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )

    return PlainTextResponse(content=text, media_type="text/plain")


@router.get(
    "/export/stats",
    response_model=ExportStatsResponse,
    responses={
        200: {"description": "Export statistics retrieved successfully"},
    },
)
async def get_export_stats() -> ExportStatsResponse:
    """Get statistics about stored behaviors.

    Returns information about the behavior history including current size,
    total behaviors recorded, and unique domains/agent types.

    Returns:
        ExportStatsResponse with statistics.
    """
    history_store = _get_history_store()
    stats = history_store.get_stats()

    return ExportStatsResponse(
        size=stats["size"],
        total_recorded=stats["total_recorded"],
        max_size=stats["max_size"],
        unique_domains=stats["unique_domains"],
        unique_agent_types=stats["unique_agent_types"],
    )


@router.get(
    "/export/domains",
    response_model=list[str],
    responses={
        200: {"description": "List of unique domain IDs"},
    },
)
async def get_export_domains() -> list[str]:
    """Get list of unique domain IDs in the behavior history.

    Useful for discovering available domains for filtering.

    Returns:
        List of unique domain IDs.
    """
    history_store = _get_history_store()
    return history_store.get_domains()


@router.get(
    "/export/agent-types",
    response_model=list[str],
    responses={
        200: {"description": "List of unique agent types"},
    },
)
async def get_export_agent_types(
    domain_id: str | None = Query(default=None, description="Optional domain ID filter"),
) -> list[str]:
    """Get list of unique agent types in the behavior history.

    Optionally filter by domain ID.

    Args:
        domain_id: Optional domain ID to filter by.

    Returns:
        List of unique agent types.
    """
    history_store = _get_history_store()
    return history_store.get_agent_types(domain_id=domain_id)


@router.delete(
    "/export",
    responses={
        200: {"description": "Behavior history cleared successfully"},
    },
)
async def clear_behavior_history() -> dict[str, str]:
    """Clear all stored behavior history.

    This permanently removes all stored behaviors. Use with caution.

    Returns:
        Success message.
    """
    history_store = _get_history_store()
    history_store.clear()
    logger.info("Behavior history cleared via API")
    return {"message": "Behavior history cleared successfully"}


# ============================================================================
# Cache Inspection Endpoints
# ============================================================================


class CachedBehaviorEntry(BaseModel):
    """A single cached behavior entry.

    Attributes:
        key: The cache key (format: domain_id:agent_type:context_hash).
        action: The cached action.
        parameters: The cached action parameters.
        reasoning: The cached reasoning.
        created_at: When the entry was cached (epoch seconds).
        expires_at: When the entry expires (epoch seconds).
        ttl_remaining: Seconds until expiration.
    """

    key: str = Field(description="Cache key")
    action: str = Field(description="Cached action")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    reasoning: str = Field(default="", description="Cached reasoning")
    created_at: float = Field(description="When entry was cached (epoch)")
    expires_at: float = Field(description="When entry expires (epoch)")
    ttl_remaining: float = Field(description="Seconds until expiration")


class CacheStatsInfo(BaseModel):
    """Cache statistics information.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        hit_rate: Cache hit rate (0.0-1.0).
        evictions: Number of LRU evictions.
        expirations: Number of TTL expirations.
    """

    hits: int = Field(description="Number of cache hits")
    misses: int = Field(description="Number of cache misses")
    hit_rate: float = Field(description="Cache hit rate (0.0-1.0)")
    evictions: int = Field(description="Number of LRU evictions")
    expirations: int = Field(description="Number of TTL expirations")


class CacheResponse(BaseModel):
    """Response for cache listing endpoint.

    Attributes:
        entries: List of cached behavior entries.
        total_entries: Total number of entries in the cache.
        stats: Cache statistics.
    """

    entries: list[CachedBehaviorEntry] = Field(description="Cached behavior entries")
    total_entries: int = Field(description="Total entries in cache")
    stats: CacheStatsInfo = Field(description="Cache statistics")


@router.get(
    "/cache",
    response_model=CacheResponse,
    responses={
        200: {"description": "Cache entries listed successfully"},
    },
)
async def get_cache(
    domain_id: str | None = Query(default=None, description="Filter by domain ID"),
) -> CacheResponse:
    """List cached behaviors for inspection and debugging.

    Returns all cached behavior entries, optionally filtered by domain.
    Also includes cache statistics (hits, misses, hit rate).

    Args:
        domain_id: Optional domain ID to filter entries.

    Returns:
        CacheResponse with list of entries and cache stats.
    """
    import time as time_module

    cache = _get_behavior_cache()
    now = time_module.time()

    # Get entries, optionally filtered by domain
    raw_entries = cache.list_entries(domain_id=domain_id)

    # Convert to response model
    entries = [
        CachedBehaviorEntry(
            key=key,
            action=entry.behavior.action,
            parameters=entry.behavior.parameters,
            reasoning=entry.behavior.reasoning,
            created_at=entry.created_at,
            expires_at=entry.expires_at,
            ttl_remaining=max(0.0, entry.expires_at - now),
        )
        for key, entry in raw_entries
    ]

    # Get cache stats
    stats_dict = cache.get_stats()
    stats = CacheStatsInfo(
        hits=stats_dict["hits"],
        misses=stats_dict["misses"],
        hit_rate=stats_dict["hit_rate"],
        evictions=stats_dict["evictions"],
        expirations=stats_dict["expirations"],
    )

    logger.debug(
        "Listed %d cache entries (domain_id=%s)",
        len(entries),
        domain_id or "all",
    )

    return CacheResponse(
        entries=entries,
        total_entries=stats_dict["size"],
        stats=stats,
    )

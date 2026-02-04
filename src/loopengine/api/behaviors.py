"""API endpoints for behavior generation.

This module provides REST API endpoints for generating agent behaviors
using the AIBehaviorEngine and domain configurations.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from loopengine.behaviors import (
    AIBehaviorEngine,
    AIBehaviorEngineError,
    DomainStore,
    DomainStoreError,
)
from loopengine.behaviors.prompt_builder import AgentContext, DomainContext

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
    domain_context = DomainContext(
        domain_type=stored.schema_.domain_type,
        domain_description=stored.schema_.description,
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

"""API endpoints for domain configuration management.

This module provides REST API endpoints for creating, retrieving, and managing
domain configurations via DomainParser and DomainStore.
"""

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from loopengine.behaviors import (
    DomainParser,
    DomainParserError,
    DomainSchema,
    DomainStore,
    DomainStoreError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/domains", tags=["domains"])


class CreateDomainRequest(BaseModel):
    """Request body for creating a domain."""

    description: str = Field(
        description="Natural language description of the domain",
        min_length=1,
    )
    domain_id: str | None = Field(
        default=None,
        description="Optional custom domain ID. If not provided, auto-generated from domain type.",
    )

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        """Validate description is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Description cannot be empty or whitespace only")
        return v.strip()

    @field_validator("domain_id")
    @classmethod
    def validate_domain_id(cls, v: str | None) -> str | None:
        """Validate domain_id format if provided."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if not all(c.isalnum() or c in ("_", "-") for c in v):
            raise ValueError(
                "Domain ID can only contain alphanumeric characters, underscores, and hyphens"
            )
        return v


class CreateDomainResponse(BaseModel):
    """Response body for domain creation."""

    domain_id: str = Field(description="Unique identifier for the created domain")
    schema_: DomainSchema = Field(alias="schema", description="Extracted domain schema")
    metadata: dict[str, Any] = Field(description="Domain metadata including version")

    model_config = {"populate_by_name": True}


def _generate_domain_id(domain_type: str) -> str:
    """Generate a valid domain ID from the domain type.

    Args:
        domain_type: The domain type string (e.g., "sandwich shop").

    Returns:
        A valid domain ID (e.g., "sandwich_shop").
    """
    # Convert to lowercase and replace spaces/special chars with underscore
    domain_id = domain_type.lower()
    domain_id = re.sub(r"[^a-z0-9]+", "_", domain_id)
    # Remove leading/trailing underscores
    domain_id = domain_id.strip("_")
    # Ensure not empty
    if not domain_id:
        domain_id = "domain"
    return domain_id


def _get_parser() -> DomainParser:
    """Get or create a DomainParser instance.

    Returns:
        DomainParser instance.
    """
    return DomainParser()


def _get_store() -> DomainStore:
    """Get or create a DomainStore instance.

    Returns:
        DomainStore instance.
    """
    return DomainStore()


@router.post(
    "",
    response_model=CreateDomainResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Domain created successfully"},
        200: {"description": "Domain updated successfully"},
        400: {"description": "Invalid request - empty description or invalid domain ID"},
        500: {"description": "Internal error - parsing or storage failed"},
    },
)
async def create_domain(request: CreateDomainRequest) -> CreateDomainResponse:
    """Create or update a domain configuration from a natural language description.

    Parses the description using AI to extract structured domain schema including
    agent types, resources, and interactions. Stores the result and returns the
    domain ID and schema.

    Args:
        request: Request containing description and optional domain_id.

    Returns:
        CreateDomainResponse with domain_id, extracted schema, and metadata.

    Raises:
        HTTPException: 400 if description is invalid, 500 if parsing/storage fails.
    """
    parser = _get_parser()
    store = _get_store()

    # Parse the description
    try:
        schema, parse_metadata = parser.parse_with_metadata(request.description)
    except DomainParserError as e:
        logger.warning("Domain parsing failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse domain description: {e}",
        ) from e
    except Exception as e:
        logger.error("Unexpected error during domain parsing: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during domain parsing",
        ) from e

    # Determine domain ID
    domain_id = request.domain_id or _generate_domain_id(schema.domain_type)

    # Store the domain
    try:
        stored = store.save(domain_id, schema)
    except DomainStoreError as e:
        logger.error("Domain storage failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store domain: {e}",
        ) from e

    # Build response metadata
    response_metadata = {
        "domain_id": stored.metadata.domain_id,
        "version": stored.metadata.version,
        "created_at": stored.metadata.created_at,
        "updated_at": stored.metadata.updated_at,
        "parse_latency_ms": parse_metadata.get("latency_ms"),
        "agent_types_extracted": parse_metadata.get("agent_types_extracted"),
        "resources_extracted": parse_metadata.get("resources_extracted"),
        "interactions_extracted": parse_metadata.get("interactions_extracted"),
    }

    logger.info(
        "Domain %s created/updated (version %d) with %d agent types",
        domain_id,
        stored.metadata.version,
        len(schema.agent_types),
    )

    return CreateDomainResponse(
        domain_id=domain_id,
        schema=schema,
        metadata=response_metadata,
    )

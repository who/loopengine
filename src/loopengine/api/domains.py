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
    ConstraintSchema,
    DomainMetadata,
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


class GetDomainResponse(BaseModel):
    """Response body for retrieving a domain."""

    domain_id: str = Field(description="Unique identifier for the domain")
    schema_: DomainSchema = Field(alias="schema", description="Domain schema")
    metadata: DomainMetadata = Field(description="Domain metadata including version and timestamps")

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


@router.get(
    "/{domain_id}",
    response_model=GetDomainResponse,
    responses={
        200: {"description": "Domain retrieved successfully"},
        404: {"description": "Domain not found"},
    },
)
async def get_domain(domain_id: str) -> GetDomainResponse:
    """Retrieve a domain configuration by ID.

    Args:
        domain_id: Unique identifier for the domain.

    Returns:
        GetDomainResponse with domain schema and metadata.

    Raises:
        HTTPException: 404 if domain doesn't exist.
    """
    store = _get_store()

    try:
        stored = store.load(domain_id)
    except DomainStoreError as e:
        logger.warning("Domain not found: %s", domain_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{domain_id}' not found",
        ) from e

    logger.info("Retrieved domain %s (version %d)", domain_id, stored.metadata.version)

    return GetDomainResponse(
        domain_id=domain_id,
        schema=stored.schema_,
        metadata=stored.metadata,
    )


class AddConstraintRequest(BaseModel):
    """Request body for adding a constraint to a domain."""

    text: str = Field(
        description="The constraint text in natural language",
        min_length=1,
    )
    constraint_type: str = Field(
        default="positive",
        description="Type of constraint: 'positive' (always do X) or 'negative' (never do Y)",
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Validate text is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Constraint text cannot be empty or whitespace only")
        return v.strip()

    @field_validator("constraint_type")
    @classmethod
    def validate_constraint_type(cls, v: str) -> str:
        """Validate constraint_type is valid."""
        v = v.lower().strip()
        if v not in ("positive", "negative"):
            raise ValueError("constraint_type must be 'positive' or 'negative'")
        return v


class UpdateConstraintsRequest(BaseModel):
    """Request body for updating all constraints on a domain."""

    constraints: list[AddConstraintRequest] = Field(
        description="List of constraints to set on the domain"
    )


class ConstraintResponse(BaseModel):
    """Response for constraint operations."""

    domain_id: str = Field(description="Domain ID")
    constraints: list[ConstraintSchema] = Field(description="Current constraints")
    version: int = Field(description="Updated domain version")


@router.post(
    "/{domain_id}/constraints",
    response_model=ConstraintResponse,
    responses={
        200: {"description": "Constraint added successfully"},
        404: {"description": "Domain not found"},
    },
)
async def add_constraint(
    domain_id: str,
    request: AddConstraintRequest,
) -> ConstraintResponse:
    """Add a behavioral constraint to an existing domain.

    Constraints can be added without re-parsing the domain description.
    This allows immediate effect on behavior generation.

    Args:
        domain_id: ID of the domain to add constraint to.
        request: Constraint details.

    Returns:
        ConstraintResponse with updated constraints list.

    Raises:
        HTTPException: 404 if domain doesn't exist.
    """
    store = _get_store()

    # Load existing domain
    try:
        stored = store.load(domain_id)
    except DomainStoreError as e:
        logger.warning("Domain not found for constraint add: %s", domain_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{domain_id}' not found",
        ) from e

    # Add the new constraint
    new_constraint = ConstraintSchema(
        text=request.text,
        constraint_type=request.constraint_type,
    )
    updated_constraints = [*stored.schema_.constraints, new_constraint]

    # Create updated schema
    updated_schema = DomainSchema(
        domain_type=stored.schema_.domain_type,
        description=stored.schema_.description,
        agent_types=stored.schema_.agent_types,
        resources=stored.schema_.resources,
        interactions=stored.schema_.interactions,
        constraints=updated_constraints,
    )

    # Save updated domain
    try:
        updated = store.save(domain_id, updated_schema)
    except DomainStoreError as e:
        logger.error("Failed to save constraint: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save constraint: {e}",
        ) from e

    logger.info(
        "Added constraint to domain %s (version %d): %s",
        domain_id,
        updated.metadata.version,
        request.text[:50],
    )

    return ConstraintResponse(
        domain_id=domain_id,
        constraints=updated_schema.constraints,
        version=updated.metadata.version,
    )


@router.put(
    "/{domain_id}/constraints",
    response_model=ConstraintResponse,
    responses={
        200: {"description": "Constraints updated successfully"},
        404: {"description": "Domain not found"},
    },
)
async def update_constraints(
    domain_id: str,
    request: UpdateConstraintsRequest,
) -> ConstraintResponse:
    """Replace all constraints on a domain.

    This replaces the entire constraints list, allowing modification
    or removal of constraints without re-parsing the domain.

    Args:
        domain_id: ID of the domain to update.
        request: New constraints list.

    Returns:
        ConstraintResponse with updated constraints list.

    Raises:
        HTTPException: 404 if domain doesn't exist.
    """
    store = _get_store()

    # Load existing domain
    try:
        stored = store.load(domain_id)
    except DomainStoreError as e:
        logger.warning("Domain not found for constraint update: %s", domain_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{domain_id}' not found",
        ) from e

    # Convert request constraints to schema constraints
    updated_constraints = [
        ConstraintSchema(text=c.text, constraint_type=c.constraint_type)
        for c in request.constraints
    ]

    # Create updated schema
    updated_schema = DomainSchema(
        domain_type=stored.schema_.domain_type,
        description=stored.schema_.description,
        agent_types=stored.schema_.agent_types,
        resources=stored.schema_.resources,
        interactions=stored.schema_.interactions,
        constraints=updated_constraints,
    )

    # Save updated domain
    try:
        updated = store.save(domain_id, updated_schema)
    except DomainStoreError as e:
        logger.error("Failed to update constraints: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update constraints: {e}",
        ) from e

    logger.info(
        "Updated constraints on domain %s (version %d): %d constraints",
        domain_id,
        updated.metadata.version,
        len(updated_constraints),
    )

    return ConstraintResponse(
        domain_id=domain_id,
        constraints=updated_schema.constraints,
        version=updated.metadata.version,
    )


@router.get(
    "/{domain_id}/constraints",
    response_model=ConstraintResponse,
    responses={
        200: {"description": "Constraints retrieved successfully"},
        404: {"description": "Domain not found"},
    },
)
async def get_constraints(domain_id: str) -> ConstraintResponse:
    """Get constraints for a domain.

    Args:
        domain_id: ID of the domain.

    Returns:
        ConstraintResponse with current constraints list.

    Raises:
        HTTPException: 404 if domain doesn't exist.
    """
    store = _get_store()

    try:
        stored = store.load(domain_id)
    except DomainStoreError as e:
        logger.warning("Domain not found for constraint get: %s", domain_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{domain_id}' not found",
        ) from e

    return ConstraintResponse(
        domain_id=domain_id,
        constraints=stored.schema_.constraints,
        version=stored.metadata.version,
    )


@router.delete(
    "/{domain_id}/constraints/{constraint_index}",
    response_model=ConstraintResponse,
    responses={
        200: {"description": "Constraint deleted successfully"},
        404: {"description": "Domain or constraint not found"},
    },
)
async def delete_constraint(
    domain_id: str,
    constraint_index: int,
) -> ConstraintResponse:
    """Delete a specific constraint from a domain by index.

    Args:
        domain_id: ID of the domain.
        constraint_index: Zero-based index of constraint to delete.

    Returns:
        ConstraintResponse with updated constraints list.

    Raises:
        HTTPException: 404 if domain or constraint doesn't exist.
    """
    store = _get_store()

    # Load existing domain
    try:
        stored = store.load(domain_id)
    except DomainStoreError as e:
        logger.warning("Domain not found for constraint delete: %s", domain_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain '{domain_id}' not found",
        ) from e

    # Validate index
    if constraint_index < 0 or constraint_index >= len(stored.schema_.constraints):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Constraint index {constraint_index} not found. "
            f"Domain has {len(stored.schema_.constraints)} constraints.",
        )

    # Remove the constraint
    updated_constraints = list(stored.schema_.constraints)
    removed = updated_constraints.pop(constraint_index)

    # Create updated schema
    updated_schema = DomainSchema(
        domain_type=stored.schema_.domain_type,
        description=stored.schema_.description,
        agent_types=stored.schema_.agent_types,
        resources=stored.schema_.resources,
        interactions=stored.schema_.interactions,
        constraints=updated_constraints,
    )

    # Save updated domain
    try:
        updated = store.save(domain_id, updated_schema)
    except DomainStoreError as e:
        logger.error("Failed to delete constraint: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete constraint: {e}",
        ) from e

    logger.info(
        "Deleted constraint from domain %s (version %d): %s",
        domain_id,
        updated.metadata.version,
        removed.text[:50],
    )

    return ConstraintResponse(
        domain_id=domain_id,
        constraints=updated_schema.constraints,
        version=updated.metadata.version,
    )

"""Domain description parser for extracting structured schema from natural language.

This module provides the DomainParser class for parsing natural language domain
descriptions into structured DomainSchema objects using LLM-based extraction.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

from loopengine.behaviors.config import LLMConfig, get_llm_config
from loopengine.behaviors.llm_client import LLMClient, LLMQuery
from loopengine.behaviors.providers.claude import ClaudeClient

logger = logging.getLogger(__name__)


class AgentTypeSchema(BaseModel):
    """Schema for an agent type in a domain.

    Attributes:
        name: The name/identifier of the agent type.
        role: Description of the agent's role in the domain.
        capabilities: List of actions this agent can perform.
    """

    name: str = Field(description="Name of the agent type (e.g., 'florist', 'customer')")
    role: str = Field(default="", description="Description of the agent's role")
    capabilities: list[str] = Field(
        default_factory=list, description="Actions this agent can perform"
    )


class ResourceSchema(BaseModel):
    """Schema for a resource in a domain.

    Attributes:
        name: The name/identifier of the resource.
        description: Description of the resource.
        consumable: Whether the resource is consumed when used.
    """

    name: str = Field(description="Name of the resource")
    description: str = Field(default="", description="Description of the resource")
    consumable: bool = Field(default=True, description="Whether resource is consumed on use")


class InteractionSchema(BaseModel):
    """Schema for an interaction between agents/resources in a domain.

    Attributes:
        name: The name of the interaction.
        participants: Agent types involved in this interaction.
        description: Description of the interaction.
    """

    name: str = Field(description="Name of the interaction")
    participants: list[str] = Field(default_factory=list, description="Agent types involved")
    description: str = Field(default="", description="Description of the interaction")


class DomainSchema(BaseModel):
    """Structured schema extracted from a domain description.

    Represents the complete structure of a domain including agent types,
    resources, and interactions.

    Attributes:
        domain_type: The type/name of the domain (e.g., 'sandwich shop').
        description: A summary description of the domain.
        agent_types: List of agent types in the domain.
        resources: List of resources in the domain.
        interactions: List of interactions between agents/resources.
    """

    domain_type: str = Field(description="Type/name of the domain")
    description: str = Field(default="", description="Summary description of the domain")
    agent_types: list[AgentTypeSchema] = Field(
        default_factory=list, description="Agent types in the domain"
    )
    resources: list[ResourceSchema] = Field(
        default_factory=list, description="Resources in the domain"
    )
    interactions: list[InteractionSchema] = Field(
        default_factory=list, description="Interactions in the domain"
    )

    @field_validator("domain_type")
    @classmethod
    def domain_type_not_empty(cls, v: str) -> str:
        """Validate that domain_type is not empty."""
        if not v or not v.strip():
            raise ValueError("domain_type cannot be empty")
        return v.strip()


class DomainParserError(Exception):
    """Exception raised when domain parsing fails."""

    pass


class DomainParser:
    """Parser that extracts structured domain schema from natural language.

    Uses an LLM to interpret natural language domain descriptions and
    extract structured schema including agent types, resources, and
    interactions.

    Example:
        >>> parser = DomainParser()
        >>> schema = parser.parse("A sandwich shop where employees make sandwiches for customers")
        >>> print(schema.domain_type)
        'sandwich shop'
        >>> print([a.name for a in schema.agent_types])
        ['sandwich_maker', 'customer']
    """

    SYSTEM_MESSAGE = """You are a domain schema extractor. Your task is to analyze a natural \
language description of a domain (business, system, or scenario) and extract a structured schema.

You MUST respond with valid JSON in this exact format:
{
    "domain_type": "short name for the domain",
    "description": "brief summary of the domain",
    "agent_types": [
        {
            "name": "agent_type_name",
            "role": "description of what this agent does",
            "capabilities": ["action1", "action2"]
        }
    ],
    "resources": [
        {
            "name": "resource_name",
            "description": "what this resource is",
            "consumable": true
        }
    ],
    "interactions": [
        {
            "name": "interaction_name",
            "participants": ["agent_type1", "agent_type2"],
            "description": "what happens in this interaction"
        }
    ]
}

Guidelines:
- Extract agent types based on roles mentioned or implied in the description
- Use snake_case for agent type names and resource names
- List capabilities as verbs (e.g., "make_sandwich", "take_order", "wait")
- Identify resources that agents create, consume, or exchange
- Identify interactions between different agent types
- If the description is vague, make reasonable inferences
- Always include at least one agent type"""

    PROMPT_TEMPLATE = """Analyze the following domain description and extract a structured schema.

Domain Description:
{description}

Remember to respond with ONLY valid JSON matching the required format."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        """Initialize the domain parser.

        Args:
            llm_client: LLM client to use for parsing. If not provided,
                creates one based on configuration.
            config: LLM configuration. If not provided, loads from environment.
        """
        self._config = config or get_llm_config()
        self._llm_client = llm_client or self._create_llm_client()

    def _create_llm_client(self) -> LLMClient:
        """Create an LLM client based on configuration.

        Returns:
            Configured LLM client.
        """
        return ClaudeClient(self._config)

    def parse(self, description: str) -> DomainSchema:
        """Parse a natural language domain description into a structured schema.

        Args:
            description: Natural language description of the domain.
                Can range from a single sentence to a paragraph.

        Returns:
            DomainSchema with extracted agent types, resources, and interactions.

        Raises:
            DomainParserError: If the description is invalid or parsing fails.
        """
        if not description or not description.strip():
            raise DomainParserError("Description cannot be empty")

        description = description.strip()

        # Build the query
        prompt = self.PROMPT_TEMPLATE.format(description=description)
        query = LLMQuery(
            prompt=prompt,
            system_message=self.SYSTEM_MESSAGE,
        )

        try:
            logger.debug("Parsing domain description: %s...", description[:50])

            # Query the LLM
            response = self._llm_client.query(query)

            # Parse the response
            return self._parse_response(response.reasoning or response.action, description)

        except DomainParserError:
            raise
        except Exception as e:
            logger.error("Domain parsing failed: %s", str(e))
            raise DomainParserError(f"Failed to parse domain description: {e}") from e

    def _parse_response(self, raw_response: str, original_description: str) -> DomainSchema:
        """Parse the LLM response into a DomainSchema.

        Args:
            raw_response: Raw text response from the LLM.
            original_description: Original description for error context.

        Returns:
            Parsed DomainSchema.

        Raises:
            DomainParserError: If parsing fails.
        """
        json_text = self._extract_json(raw_response)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", raw_response[:200])
            raise DomainParserError(f"Invalid JSON in response: {e}") from e

        if not isinstance(data, dict):
            raise DomainParserError(f"Expected JSON object, got {type(data).__name__}")

        try:
            return DomainSchema(**data)
        except Exception as e:
            raise DomainParserError(f"Failed to validate schema: {e}") from e

    def _extract_json(self, text: str) -> str:
        """Extract JSON from potentially wrapped text.

        Handles:
        - Raw JSON
        - ```json ... ``` blocks
        - ``` ... ``` blocks

        Args:
            text: Raw text that may contain JSON.

        Returns:
            Extracted JSON string.
        """
        import re

        # Try markdown json code block first
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Try generic code block
        if "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted.startswith("{") or extracted.startswith("["):
                    return extracted

        # Try to find JSON object by matching balanced braces
        json_str = self._find_balanced_json(text)
        if json_str:
            return json_str

        return text.strip()

    def _find_balanced_json(self, text: str) -> str | None:
        """Find balanced JSON object in text.

        Args:
            text: Text to search.

        Returns:
            Extracted JSON string or None if not found.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i, char in enumerate(text[start:], start):
            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"' and not escape:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None

    def parse_with_metadata(self, description: str) -> tuple[DomainSchema, dict[str, Any]]:
        """Parse a domain description and return schema with metadata.

        Useful when you want additional context about the parsing process.

        Args:
            description: Natural language description of the domain.

        Returns:
            Tuple of (DomainSchema, metadata dict with provider and timing info).

        Raises:
            DomainParserError: If parsing fails.
        """
        import time

        start_time = time.perf_counter()
        schema = self.parse(description)
        latency_ms = (time.perf_counter() - start_time) * 1000

        metadata = {
            "latency_ms": round(latency_ms, 2),
            "input_length": len(description),
            "agent_types_extracted": len(schema.agent_types),
            "resources_extracted": len(schema.resources),
            "interactions_extracted": len(schema.interactions),
        }

        return schema, metadata

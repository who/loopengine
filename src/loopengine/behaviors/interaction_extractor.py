"""Interaction extractor for extracting interaction patterns from domain descriptions.

This module provides the InteractionExtractor class for extracting valid
agent-agent and agent-resource interactions from natural language domain
descriptions.
"""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from loopengine.behaviors.config import LLMConfig, get_llm_config
from loopengine.behaviors.llm_client import LLMClient, LLMQuery
from loopengine.behaviors.providers.claude import ClaudeClient

logger = logging.getLogger(__name__)


class Interaction(BaseModel):
    """Detailed interaction model extracted from domain description.

    Represents a valid interaction pattern between agents or between
    agents and resources.

    Attributes:
        name: Unique identifier for the interaction (snake_case).
        source_agent: Agent type that initiates the interaction.
        target: Target of interaction (agent type or resource name).
        action_type: Type of action (request, provide, consume, transfer, etc.).
        conditions: List of conditions that must be true for interaction.
        description: Description of the interaction.
    """

    name: str = Field(description="Unique identifier for the interaction (snake_case)")
    source_agent: str = Field(description="Agent type that initiates the interaction")
    target: str = Field(description="Target of interaction (agent type or resource)")
    action_type: str = Field(
        default="interact",
        description="Type of action: request, provide, consume, transfer, query, update",
    )
    conditions: list[str] = Field(
        default_factory=list, description="Conditions for the interaction"
    )
    description: str = Field(default="", description="Description of the interaction")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is not empty and is properly formatted."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("source_agent")
    @classmethod
    def validate_source_agent(cls, v: str) -> str:
        """Validate source agent is not empty."""
        if not v or not v.strip():
            raise ValueError("source_agent cannot be empty")
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        """Validate target is not empty."""
        if not v or not v.strip():
            raise ValueError("target cannot be empty")
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        """Normalize action type."""
        return v.strip().lower().replace(" ", "_").replace("-", "_")


class InteractionExtractorError(Exception):
    """Exception raised when interaction extraction fails."""

    pass


class InteractionExtractor:
    """Extractor that identifies and extracts interactions from domain descriptions.

    Uses an LLM to interpret natural language domain descriptions and
    extract valid agent-agent and agent-resource interaction patterns.

    Example:
        >>> extractor = InteractionExtractor()
        >>> interactions = extractor.extract("Customers order from florists")
        >>> print([i.name for i in interactions])
        ['customer_order']
        >>> print(interactions[0].source_agent)
        'customer'
        >>> print(interactions[0].target)
        'florist'
    """

    SYSTEM_MESSAGE = """You are an interaction pattern extractor. Your task is to analyze \
a natural language description of a domain and extract all valid interactions between \
agents and resources.

Interactions fall into these categories:
1. AGENT-AGENT - Direct communication or exchange between agents
   (e.g., customer orders from florist, manager assigns task to worker)
2. AGENT-RESOURCE - Agent manipulating, consuming, or creating resources
   (e.g., worker consumes materials, chef creates dish from ingredients)

Action types:
- request: Asking for something (e.g., customer requests order)
- provide: Giving something (e.g., worker provides service)
- consume: Using up a resource (e.g., chef consumes ingredients)
- transfer: Moving something between agents (e.g., delivery transfers package)
- query: Asking for information (e.g., customer queries availability)
- update: Modifying state (e.g., manager updates schedule)
- create: Making something new (e.g., baker creates bread)

You MUST respond with valid JSON in this exact format:
{
    "interactions": [
        {
            "name": "interaction_name",
            "source_agent": "agent_type_that_initiates",
            "target": "target_agent_or_resource",
            "action_type": "request|provide|consume|transfer|query|update|create",
            "conditions": ["condition1", "condition2"],
            "description": "what happens in this interaction"
        }
    ]
}

Guidelines:
- Use snake_case for all names
- source_agent is always an agent type (the initiator)
- target can be an agent type OR a resource name
- Identify conditions:
  - "when available" -> "target_available"
  - "if in stock" -> "sufficient_inventory"
  - "during hours" -> "within_operating_hours"
  - "with permission" -> "has_permission"
- Extract both explicit and implied interactions
- Every domain should have at least one interaction"""

    PROMPT_TEMPLATE = """Analyze the following domain description and extract all interactions.

Domain Description:
{description}

Extract agent-agent and agent-resource interaction patterns.

Remember to respond with ONLY valid JSON matching the required format."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        """Initialize the interaction extractor.

        Args:
            llm_client: LLM client to use for extraction. If not provided,
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

    def extract(self, description: str) -> list[Interaction]:
        """Extract interactions from a natural language domain description.

        Args:
            description: Natural language description of the domain.
                Can range from a single sentence to a paragraph.

        Returns:
            List of Interaction objects with complete definitions.

        Raises:
            InteractionExtractorError: If the description is invalid or extraction fails.
        """
        if not description or not description.strip():
            raise InteractionExtractorError("Description cannot be empty")

        description = description.strip()

        prompt = self.PROMPT_TEMPLATE.format(description=description)
        query = LLMQuery(
            prompt=prompt,
            system_message=self.SYSTEM_MESSAGE,
        )

        try:
            logger.debug("Extracting interactions from: %s...", description[:50])

            response = self._llm_client.query(query)

            raw_interactions = self._parse_response(response.reasoning or response.action)

            return self._enhance_interactions(raw_interactions)

        except InteractionExtractorError:
            raise
        except Exception as e:
            logger.error("Interaction extraction failed: %s", str(e))
            raise InteractionExtractorError(f"Failed to extract interactions: {e}") from e

    def _parse_response(self, raw_response: str) -> list[dict[str, Any]]:
        """Parse the LLM response into raw interaction data.

        Args:
            raw_response: Raw text response from the LLM.

        Returns:
            List of interaction dictionaries.

        Raises:
            InteractionExtractorError: If parsing fails.
        """
        json_text = self._extract_json(raw_response)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", raw_response[:200])
            raise InteractionExtractorError(f"Invalid JSON in response: {e}") from e

        if not isinstance(data, dict):
            raise InteractionExtractorError(f"Expected JSON object, got {type(data).__name__}")

        interactions = data.get("interactions", [])
        if not isinstance(interactions, list):
            raise InteractionExtractorError("interactions must be a list")

        return interactions

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
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()

        if "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted.startswith("{") or extracted.startswith("["):
                    return extracted

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

    def _enhance_interactions(self, raw_interactions: list[dict[str, Any]]) -> list[Interaction]:
        """Enhance interaction data with defaults and validate.

        Args:
            raw_interactions: List of raw interaction dictionaries from LLM.

        Returns:
            List of validated Interaction objects.
        """
        enhanced: list[Interaction] = []

        for raw in raw_interactions:
            name = raw.get("name", "")
            source_agent = raw.get("source_agent", "")
            target = raw.get("target", "")

            if not name or not source_agent or not target:
                continue

            action_type = raw.get("action_type", "interact")
            conditions = raw.get("conditions", [])
            description = raw.get("description", "")

            if not isinstance(conditions, list):
                conditions = [conditions] if conditions else []

            try:
                interaction = Interaction(
                    name=name,
                    source_agent=source_agent,
                    target=target,
                    action_type=action_type,
                    conditions=conditions,
                    description=description,
                )
                enhanced.append(interaction)
            except ValueError as e:
                logger.warning("Skipping invalid interaction %s: %s", name, e)
                continue

        return enhanced

    def extract_with_metadata(self, description: str) -> tuple[list[Interaction], dict[str, Any]]:
        """Extract interactions and return with metadata.

        Useful when you want additional context about the extraction process.

        Args:
            description: Natural language description of the domain.

        Returns:
            Tuple of (list of Interaction, metadata dict with timing info).

        Raises:
            InteractionExtractorError: If extraction fails.
        """
        import time

        start_time = time.perf_counter()
        interactions = self.extract(description)
        latency_ms = (time.perf_counter() - start_time) * 1000

        action_counts: dict[str, int] = {}
        for i in interactions:
            action_counts[i.action_type] = action_counts.get(i.action_type, 0) + 1

        unique_agents = set()
        unique_targets = set()
        for i in interactions:
            unique_agents.add(i.source_agent)
            unique_targets.add(i.target)

        metadata = {
            "latency_ms": round(latency_ms, 2),
            "input_length": len(description),
            "interactions_extracted": len(interactions),
            "action_type_breakdown": action_counts,
            "unique_source_agents": len(unique_agents),
            "unique_targets": len(unique_targets),
            "total_conditions": sum(len(i.conditions) for i in interactions),
        }

        return interactions, metadata

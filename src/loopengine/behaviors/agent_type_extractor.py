"""Agent type extractor for extracting agent types from domain descriptions.

This module provides the AgentTypeExtractor class for extracting detailed agent
type definitions from natural language domain descriptions, including roles,
capabilities, and initial states.
"""

import json
import logging
from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator

from loopengine.behaviors.config import LLMConfig, get_llm_config
from loopengine.behaviors.llm_client import LLMClient, LLMQuery
from loopengine.behaviors.providers.claude import ClaudeClient

logger = logging.getLogger(__name__)


class AgentType(BaseModel):
    """Detailed agent type model extracted from domain description.

    Represents a complete agent type definition with role, capabilities,
    and default state that can be used to instantiate simulation agents.

    Attributes:
        name: Unique identifier for the agent type (snake_case).
        role: Description of what this agent does in the domain.
        capabilities: List of actions this agent can perform.
        default_state: Initial state values for agents of this type.
    """

    name: str = Field(description="Unique identifier for the agent type (snake_case)")
    role: str = Field(default="", description="Description of the agent's role in the domain")
    capabilities: list[str] = Field(
        default_factory=list, description="Actions this agent can perform (verb_phrases)"
    )
    default_state: dict[str, Any] = Field(
        default_factory=dict, description="Initial state values for agents of this type"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is not empty and is properly formatted."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        # Convert to snake_case if not already
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: list[str]) -> list[str]:
        """Ensure capabilities are properly formatted."""
        return [cap.strip().lower().replace(" ", "_").replace("-", "_") for cap in v if cap.strip()]


class AgentTypeExtractorError(Exception):
    """Exception raised when agent type extraction fails."""

    pass


class AgentTypeExtractor:
    """Extractor that identifies and extracts agent types from domain descriptions.

    Uses an LLM to interpret natural language domain descriptions and
    extract detailed agent type definitions, including both explicitly
    mentioned agents and implied agents.

    Example:
        >>> extractor = AgentTypeExtractor()
        >>> agents = extractor.extract("Florists make bouquets for customers")
        >>> print([a.name for a in agents])
        ['florist', 'customer']
        >>> print(agents[0].capabilities)
        ['make_bouquet']
    """

    SYSTEM_MESSAGE = """You are an agent type extractor. Your task is to analyze a natural \
language description of a domain and extract all agent types (roles/actors) present.

You must identify:
1. EXPLICIT agents - directly mentioned roles (e.g., "florists make bouquets" -> florist)
2. IMPLIED agents - roles that must exist for described actions
   (e.g., "deliveries happen" implies delivery_driver)

You MUST respond with valid JSON in this exact format:
{
    "agent_types": [
        {
            "name": "agent_type_name",
            "role": "description of what this agent does",
            "capabilities": ["action1", "action2"],
            "default_state": {
                "status": "idle",
                "energy": 100
            },
            "extraction_type": "explicit"
        }
    ]
}

Guidelines:
- Use snake_case for names and capabilities (e.g., "sandwich_maker", "make_sandwich")
- List capabilities as verb phrases (e.g., "make_bouquet", "take_order")
- For default_state, include relevant initial values:
  - "status": typically "idle" or "ready"
  - "energy": 100 (if applicable)
  - "inventory": [] or {} (if they carry items)
  - "queue": [] (if they process requests)
  - Role-specific states (e.g., "orders_completed": 0 for workers)
- extraction_type should be "explicit" if directly mentioned, "implied" if inferred
- Generate sensible defaults even if not specified in the description
- Every domain should have at least one agent type"""

    PROMPT_TEMPLATE = """Analyze the following domain description and extract all agent types.

Domain Description:
{description}

Extract both explicitly mentioned agents and implied agents that must exist for the activities.

Remember to respond with ONLY valid JSON matching the required format."""

    # Default states by common agent categories
    DEFAULT_STATES: ClassVar[dict[str, dict[str, Any]]] = {
        "worker": {"status": "idle", "energy": 100, "tasks_completed": 0},
        "customer": {"status": "browsing", "satisfaction": 50, "orders_placed": 0},
        "service": {"status": "available", "queue": [], "capacity": 10},
        "driver": {"status": "idle", "location": "base", "deliveries_completed": 0},
        "manager": {"status": "monitoring", "decisions_made": 0},
        "default": {"status": "idle"},
    }

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        """Initialize the agent type extractor.

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

    def extract(self, description: str) -> list[AgentType]:
        """Extract agent types from a natural language domain description.

        Args:
            description: Natural language description of the domain.
                Can range from a single sentence to a paragraph.

        Returns:
            List of AgentType objects with complete definitions.

        Raises:
            AgentTypeExtractorError: If the description is invalid or extraction fails.
        """
        if not description or not description.strip():
            raise AgentTypeExtractorError("Description cannot be empty")

        description = description.strip()

        # Build the query
        prompt = self.PROMPT_TEMPLATE.format(description=description)
        query = LLMQuery(
            prompt=prompt,
            system_message=self.SYSTEM_MESSAGE,
        )

        try:
            logger.debug("Extracting agent types from: %s...", description[:50])

            # Query the LLM
            response = self._llm_client.query(query)

            # Parse the response
            raw_agents = self._parse_response(response.reasoning or response.action)

            # Enhance with defaults and validate
            return self._enhance_agent_types(raw_agents)

        except AgentTypeExtractorError:
            raise
        except Exception as e:
            logger.error("Agent type extraction failed: %s", str(e))
            raise AgentTypeExtractorError(f"Failed to extract agent types: {e}") from e

    def _parse_response(self, raw_response: str) -> list[dict[str, Any]]:
        """Parse the LLM response into raw agent type data.

        Args:
            raw_response: Raw text response from the LLM.

        Returns:
            List of agent type dictionaries.

        Raises:
            AgentTypeExtractorError: If parsing fails.
        """
        json_text = self._extract_json(raw_response)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", raw_response[:200])
            raise AgentTypeExtractorError(f"Invalid JSON in response: {e}") from e

        if not isinstance(data, dict):
            raise AgentTypeExtractorError(f"Expected JSON object, got {type(data).__name__}")

        agent_types = data.get("agent_types", [])
        if not isinstance(agent_types, list):
            raise AgentTypeExtractorError("agent_types must be a list")

        return agent_types

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

    def _enhance_agent_types(self, raw_agents: list[dict[str, Any]]) -> list[AgentType]:
        """Enhance agent type data with defaults and validate.

        Args:
            raw_agents: List of raw agent type dictionaries from LLM.

        Returns:
            List of validated AgentType objects with enhanced defaults.
        """
        enhanced: list[AgentType] = []

        for raw in raw_agents:
            # Ensure name exists
            name = raw.get("name", "")
            if not name:
                continue

            # Get role
            role = raw.get("role", "")

            # Get capabilities, filtering empty
            capabilities = [c for c in raw.get("capabilities", []) if c and c.strip()]

            # Get or generate default state
            default_state = raw.get("default_state", {})
            if not default_state:
                default_state = self._generate_default_state(name, role)

            # Create the AgentType
            try:
                agent_type = AgentType(
                    name=name,
                    role=role,
                    capabilities=capabilities,
                    default_state=default_state,
                )
                enhanced.append(agent_type)
            except ValueError as e:
                logger.warning("Skipping invalid agent type %s: %s", name, e)
                continue

        return enhanced

    def _generate_default_state(self, name: str, role: str) -> dict[str, Any]:
        """Generate sensible default state based on agent name and role.

        Args:
            name: Agent type name.
            role: Agent type role description.

        Returns:
            Dictionary of default state values.
        """
        name_lower = name.lower()
        role_lower = role.lower()

        # Check for customer-like agents
        customer_words = ["customer", "client", "buyer"]
        if any(word in name_lower or word in role_lower for word in customer_words):
            return self.DEFAULT_STATES["customer"].copy()

        # Check for driver/delivery agents
        driver_words = ["driver", "delivery", "courier"]
        if any(word in name_lower or word in role_lower for word in driver_words):
            return self.DEFAULT_STATES["driver"].copy()

        # Check for manager/supervisor agents
        manager_words = ["manager", "supervisor", "boss"]
        if any(word in name_lower or word in role_lower for word in manager_words):
            return self.DEFAULT_STATES["manager"].copy()

        # Check for service agents
        service_words = ["service", "receptionist", "host"]
        if any(word in name_lower or word in role_lower for word in service_words):
            return self.DEFAULT_STATES["service"].copy()

        # Check for worker agents (default for most roles)
        if any(
            word in name_lower or word in role_lower
            for word in ["maker", "worker", "staff", "employee", "florist", "baker", "chef"]
        ):
            return self.DEFAULT_STATES["worker"].copy()

        # Fallback to default
        return self.DEFAULT_STATES["default"].copy()

    def extract_with_metadata(self, description: str) -> tuple[list[AgentType], dict[str, Any]]:
        """Extract agent types and return with metadata.

        Useful when you want additional context about the extraction process.

        Args:
            description: Natural language description of the domain.

        Returns:
            Tuple of (list of AgentType, metadata dict with timing info).

        Raises:
            AgentTypeExtractorError: If extraction fails.
        """
        import time

        start_time = time.perf_counter()
        agent_types = self.extract(description)
        latency_ms = (time.perf_counter() - start_time) * 1000

        explicit_count = sum(1 for _ in agent_types)  # Could track extraction_type if preserved

        metadata = {
            "latency_ms": round(latency_ms, 2),
            "input_length": len(description),
            "agent_types_extracted": len(agent_types),
            "explicit_count": explicit_count,
            "total_capabilities": sum(len(a.capabilities) for a in agent_types),
        }

        return agent_types, metadata

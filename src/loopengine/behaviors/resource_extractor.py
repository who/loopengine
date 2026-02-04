"""Resource extractor for extracting resources from domain descriptions.

This module provides the ResourceExtractor class for extracting detailed resource
definitions from natural language domain descriptions, including resource types,
quantities, and constraints.
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


class Resource(BaseModel):
    """Detailed resource model extracted from domain description.

    Represents a complete resource definition with type, quantity,
    and constraints for simulation.

    Attributes:
        name: Unique identifier for the resource (snake_case).
        resource_type: Category of resource (item, location, queue, service).
        description: Description of the resource.
        initial_quantity: Starting quantity in simulation (-1 for unlimited).
        constraints: List of constraints on the resource.
    """

    name: str = Field(description="Unique identifier for the resource (snake_case)")
    resource_type: str = Field(
        default="item", description="Type of resource: item, location, queue, service"
    )
    description: str = Field(default="", description="Description of the resource")
    initial_quantity: int = Field(default=-1, description="Initial quantity (-1 for unlimited)")
    constraints: list[str] = Field(default_factory=list, description="Constraints on the resource")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is not empty and is properly formatted."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("resource_type")
    @classmethod
    def validate_resource_type(cls, v: str) -> str:
        """Validate resource type is valid."""
        valid_types = {"item", "location", "queue", "service"}
        normalized = v.strip().lower()
        if normalized not in valid_types:
            return "item"  # Default to item if unknown
        return normalized


class ResourceExtractorError(Exception):
    """Exception raised when resource extraction fails."""

    pass


class ResourceExtractor:
    """Extractor that identifies and extracts resources from domain descriptions.

    Uses an LLM to interpret natural language domain descriptions and
    extract detailed resource definitions, including types, quantities,
    and constraints.

    Example:
        >>> extractor = ResourceExtractor()
        >>> resources = extractor.extract("A flower shop with limited roses")
        >>> print([r.name for r in resources])
        ['roses', 'flowers']
        >>> print(resources[0].constraints)
        ['limited_stock']
    """

    SYSTEM_MESSAGE = """You are a resource extractor. Your task is to analyze a natural \
language description of a domain and extract all resources (items, locations, queues).

Resources include:
1. ITEMS - Physical goods that can be consumed, created, or exchanged
   (e.g., flowers, sandwiches, inventory items)
2. LOCATIONS - Places where actions occur or agents reside
   (e.g., counter, kitchen, warehouse)
3. QUEUES - Abstract resources representing waiting/processing order
   (e.g., order_queue, waiting_list)
4. SERVICES - Intangible offerings that can be consumed
   (e.g., delivery, consultation)

You MUST respond with valid JSON in this exact format:
{
    "resources": [
        {
            "name": "resource_name",
            "resource_type": "item|location|queue|service",
            "description": "what this resource is",
            "initial_quantity": 100,
            "constraints": ["limited_stock", "perishable"]
        }
    ]
}

Guidelines:
- Use snake_case for resource names
- For items, extract quantity hints (e.g., "100 roses" -> initial_quantity: 100)
- Use -1 for initial_quantity if unlimited or not specified
- Identify constraints:
  - "limited" or "scarce" -> "limited_stock"
  - "perishable" or "expires" -> "perishable"
  - "premium" or "expensive" -> "premium"
  - "shared" or "common" -> "shared"
- Extract both explicit and implied resources
- Every domain should have at least one resource"""

    PROMPT_TEMPLATE = """Analyze the following domain description and extract all resources.

Domain Description:
{description}

Extract items, locations, queues, and services mentioned or implied.

Remember to respond with ONLY valid JSON matching the required format."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        """Initialize the resource extractor.

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

    def extract(self, description: str) -> list[Resource]:
        """Extract resources from a natural language domain description.

        Args:
            description: Natural language description of the domain.
                Can range from a single sentence to a paragraph.

        Returns:
            List of Resource objects with complete definitions.

        Raises:
            ResourceExtractorError: If the description is invalid or extraction fails.
        """
        if not description or not description.strip():
            raise ResourceExtractorError("Description cannot be empty")

        description = description.strip()

        prompt = self.PROMPT_TEMPLATE.format(description=description)
        query = LLMQuery(
            prompt=prompt,
            system_message=self.SYSTEM_MESSAGE,
        )

        try:
            logger.debug("Extracting resources from: %s...", description[:50])

            response = self._llm_client.query(query)

            raw_resources = self._parse_response(response.reasoning or response.action)

            return self._enhance_resources(raw_resources)

        except ResourceExtractorError:
            raise
        except Exception as e:
            logger.error("Resource extraction failed: %s", str(e))
            raise ResourceExtractorError(f"Failed to extract resources: {e}") from e

    def _parse_response(self, raw_response: str) -> list[dict[str, Any]]:
        """Parse the LLM response into raw resource data.

        Args:
            raw_response: Raw text response from the LLM.

        Returns:
            List of resource dictionaries.

        Raises:
            ResourceExtractorError: If parsing fails.
        """
        json_text = self._extract_json(raw_response)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", raw_response[:200])
            raise ResourceExtractorError(f"Invalid JSON in response: {e}") from e

        if not isinstance(data, dict):
            raise ResourceExtractorError(f"Expected JSON object, got {type(data).__name__}")

        resources = data.get("resources", [])
        if not isinstance(resources, list):
            raise ResourceExtractorError("resources must be a list")

        return resources

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

    def _enhance_resources(self, raw_resources: list[dict[str, Any]]) -> list[Resource]:
        """Enhance resource data with defaults and validate.

        Args:
            raw_resources: List of raw resource dictionaries from LLM.

        Returns:
            List of validated Resource objects.
        """
        enhanced: list[Resource] = []

        for raw in raw_resources:
            name = raw.get("name", "")
            if not name:
                continue

            resource_type = raw.get("resource_type", "item")
            description = raw.get("description", "")
            initial_quantity = raw.get("initial_quantity", -1)
            constraints = raw.get("constraints", [])

            if not isinstance(constraints, list):
                constraints = [constraints] if constraints else []

            try:
                resource = Resource(
                    name=name,
                    resource_type=resource_type,
                    description=description,
                    initial_quantity=initial_quantity,
                    constraints=constraints,
                )
                enhanced.append(resource)
            except ValueError as e:
                logger.warning("Skipping invalid resource %s: %s", name, e)
                continue

        return enhanced

    def extract_with_metadata(self, description: str) -> tuple[list[Resource], dict[str, Any]]:
        """Extract resources and return with metadata.

        Useful when you want additional context about the extraction process.

        Args:
            description: Natural language description of the domain.

        Returns:
            Tuple of (list of Resource, metadata dict with timing info).

        Raises:
            ResourceExtractorError: If extraction fails.
        """
        import time

        start_time = time.perf_counter()
        resources = self.extract(description)
        latency_ms = (time.perf_counter() - start_time) * 1000

        type_counts: dict[str, int] = {}
        for r in resources:
            type_counts[r.resource_type] = type_counts.get(r.resource_type, 0) + 1

        metadata = {
            "latency_ms": round(latency_ms, 2),
            "input_length": len(description),
            "resources_extracted": len(resources),
            "type_breakdown": type_counts,
            "total_constraints": sum(len(r.constraints) for r in resources),
        }

        return resources, metadata

"""AI genome discovery module using Claude API.

This module implements the discover_schemas function that sends system descriptions
to the Claude API and parses GenomeSchema objects from the response.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import anthropic
from anthropic import APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from loopengine.behaviors.config import LLMConfig, get_llm_config
from loopengine.model.genome import GenomeSchema, GenomeTrait

logger = logging.getLogger(__name__)


class DiscoveryError(Exception):
    """Exception raised when genome discovery fails."""

    pass


@dataclass
class DiscoveredRole:
    """Result of discovery for a single role.

    Attributes:
        schema: The discovered GenomeSchema for this role.
        flexibility_score: Input variance expected for this role (0.0-1.0).
    """

    schema: GenomeSchema
    flexibility_score: float = 0.5


@dataclass
class DiscoveryResult:
    """Result of a full genome discovery operation.

    Attributes:
        roles: Dictionary mapping role names to their discovered schemas.
        discovery_prompt: The prompt that was sent to the AI.
        discovered_at: When the discovery was performed.
    """

    roles: dict[str, DiscoveredRole] = field(default_factory=dict)
    discovery_prompt: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)


# Discovery prompt template based on PRD Appendix B
DISCOVERY_PROMPT_TEMPLATE = """You are analyzing an organizational system to discover the meaningful
dimensions of variation for agents in each role.

SYSTEM DESCRIPTION:
{system_description_json}

For each role in this system, identify the traits that would meaningfully
affect an agent's performance. Consider:
- What cognitive, physical, social, and temperamental traits matter?
- What skills or aptitudes differentiate good from poor performance?
- What tendencies or biases affect decision-making in this role?
- What traits affect how this agent interacts with linked agents?

For each trait, provide:
- name: a snake_case identifier
- description: what this trait represents
- category: one of "physical", "cognitive", "social", "temperamental", "skill"
- min_val: minimum value (typically 0.0)
- max_val: maximum value (typically 1.0)

Also provide a flexibility_score (0.0 to 1.0) for each role indicating
how much input variance the role typically faces.

Respond with valid JSON only, no additional commentary. Use this exact structure:
{{
    "roles": {{
        "role_name": {{
            "traits": [
                {{
                    "name": "trait_name",
                    "description": "what this trait represents",
                    "category": "cognitive",
                    "min_val": 0.0,
                    "max_val": 1.0
                }}
            ],
            "flexibility_score": 0.5
        }}
    }}
}}"""


class Discoverer:
    """AI-powered genome schema discoverer.

    Uses Claude API to analyze system descriptions and discover
    meaningful genome traits for each role.

    Example:
        >>> discoverer = Discoverer()
        >>> result = discoverer.discover_schemas({
        ...     "system": "Small sandwich shop",
        ...     "roles": [{"name": "owner", "inputs": [...], "outputs": [...]}]
        ... })
        >>> print(result.roles["owner"].schema.traits)
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_RETRIES = 3
    MAX_TOKENS = 4000

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize the discoverer.

        Args:
            config: Optional LLMConfig. If not provided, loads from environment.
        """
        self._config = config or get_llm_config()
        self._client: anthropic.Anthropic | None = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Anthropic client with current configuration."""
        api_key = self._config.get_api_key()
        if not api_key:
            logger.warning("No Anthropic API key configured. Discovery will fail.")
            self._client = None
            return

        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=self._config.llm_timeout * 2,  # Discovery may take longer
            max_retries=self.MAX_RETRIES,
        )

    @retry(
        retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        """Make the API call with retry logic.

        Args:
            prompt: The formatted discovery prompt.

        Returns:
            The raw text response from Claude.

        Raises:
            DiscoveryError: If the API call fails after retries.
        """
        if not self._client:
            raise DiscoveryError(
                "Anthropic client not initialized. Check that ANTHROPIC_API_KEY is set."
            )

        logger.debug("Calling Claude API for genome discovery")

        system_msg = (
            "You are a system analyst that discovers meaningful genome traits "
            "for simulation agents. Always respond with valid JSON only."
        )
        response = self._client.messages.create(
            model=self.DEFAULT_MODEL,
            max_tokens=self.MAX_TOKENS,
            temperature=0.3,  # Lower temperature for structured output
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text content from response
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content = block.text
                break

        if not text_content:
            raise DiscoveryError("Empty response from Claude API")

        return text_content

    def _parse_response(self, text_content: str, discovery_prompt: str) -> DiscoveryResult:
        """Parse the Claude response into DiscoveryResult.

        Args:
            text_content: Raw JSON text from Claude.
            discovery_prompt: The prompt that was sent.

        Returns:
            DiscoveryResult with parsed schemas.

        Raises:
            DiscoveryError: If response cannot be parsed as valid JSON.
        """
        # Handle potential markdown code blocks
        json_text = text_content
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse discovery response as JSON: %s", text_content[:500])
            raise DiscoveryError(f"Failed to parse response as JSON: {e}") from e

        if "roles" not in data:
            raise DiscoveryError("Response missing 'roles' key")

        discovered_at = datetime.now()
        result = DiscoveryResult(
            discovery_prompt=discovery_prompt,
            discovered_at=discovered_at,
        )

        for role_name, role_data in data["roles"].items():
            traits = {}
            for trait_data in role_data.get("traits", []):
                trait = GenomeTrait(
                    name=trait_data.get("name", ""),
                    description=trait_data.get("description", ""),
                    min_val=float(trait_data.get("min_val", 0.0)),
                    max_val=float(trait_data.get("max_val", 1.0)),
                    category=trait_data.get("category", ""),
                    discovered_at=discovered_at,
                )
                traits[trait.name] = trait

            schema = GenomeSchema(
                role=role_name,
                traits=traits,
                discovered_at=discovered_at,
                discovery_prompt=discovery_prompt,
                version=1,
            )

            flexibility = float(role_data.get("flexibility_score", 0.5))
            result.roles[role_name] = DiscoveredRole(
                schema=schema,
                flexibility_score=flexibility,
            )

        return result

    def discover_schemas(self, system_description: dict[str, Any]) -> dict[str, GenomeSchema]:
        """Discover genome schemas for all roles in a system description.

        Args:
            system_description: System description dict with roles, links, etc.
                Expected format:
                {
                    "system": "Description of the system",
                    "roles": [
                        {
                            "name": "role_name",
                            "inputs": ["input1", "input2"],
                            "outputs": ["output1"],
                            "constraints": ["constraint1"],
                            "links_to": ["other_role (link_type)"]
                        }
                    ]
                }

        Returns:
            Dictionary mapping role names to their GenomeSchema.

        Raises:
            DiscoveryError: If discovery fails.
        """
        system_name = system_description.get("system", "Unknown")
        logger.info("Starting genome discovery for system: %s", system_name)

        # Format the system description as JSON for the prompt
        system_json = json.dumps(system_description, indent=2)
        prompt = DISCOVERY_PROMPT_TEMPLATE.format(system_description_json=system_json)

        try:
            text_content = self._call_api(prompt)
            result = self._parse_response(text_content, prompt)

            logger.info(
                "Discovered schemas for %d roles: %s",
                len(result.roles),
                list(result.roles.keys()),
            )

            # Extract just the schemas
            return {role: data.schema for role, data in result.roles.items()}

        except (APITimeoutError, RateLimitError) as e:
            logger.error("API error during discovery: %s", e)
            raise DiscoveryError(f"API error: {e}") from e
        except APIError as e:
            logger.error("Claude API error: %s", e.message)
            raise DiscoveryError(f"API error: {e.message}") from e
        except Exception as e:
            logger.error("Unexpected error during discovery: %s", str(e))
            raise DiscoveryError(f"Discovery failed: {e}") from e

    def discover(self, system_description: dict[str, Any]) -> DiscoveryResult:
        """Discover genome schemas with full metadata.

        This is the full discovery method that returns flexibility scores
        and other metadata in addition to the schemas.

        Args:
            system_description: System description dict.

        Returns:
            DiscoveryResult with schemas and metadata.

        Raises:
            DiscoveryError: If discovery fails.
        """
        system_name = system_description.get("system", "Unknown")
        logger.info("Starting full genome discovery for system: %s", system_name)

        system_json = json.dumps(system_description, indent=2)
        prompt = DISCOVERY_PROMPT_TEMPLATE.format(system_description_json=system_json)

        try:
            text_content = self._call_api(prompt)
            result = self._parse_response(text_content, prompt)

            logger.info(
                "Discovered schemas for %d roles: %s",
                len(result.roles),
                list(result.roles.keys()),
            )

            return result

        except (APITimeoutError, RateLimitError) as e:
            logger.error("API error during discovery: %s", e)
            raise DiscoveryError(f"API error: {e}") from e
        except APIError as e:
            logger.error("Claude API error: %s", e.message)
            raise DiscoveryError(f"API error: {e.message}") from e
        except Exception as e:
            logger.error("Unexpected error during discovery: %s", str(e))
            raise DiscoveryError(f"Discovery failed: {e}") from e


def discover_schemas(system_description: dict[str, Any]) -> dict[str, GenomeSchema]:
    """Convenience function to discover genome schemas.

    Creates a Discoverer instance and calls discover_schemas.

    Args:
        system_description: System description dict with roles, links, etc.

    Returns:
        Dictionary mapping role names to their GenomeSchema.

    Raises:
        DiscoveryError: If discovery fails.

    Example:
        >>> schemas = discover_schemas({
        ...     "system": "Small sandwich shop, 3 employees",
        ...     "roles": [
        ...         {
        ...             "name": "owner",
        ...             "inputs": ["supply_invoices", "revenue_reports"],
        ...             "outputs": ["supply_orders", "directives"],
        ...             "constraints": ["budget", "health_code"],
        ...             "links_to": ["sandwich_maker (hierarchical)"]
        ...         }
        ...     ]
        ... })
        >>> print(schemas["owner"].traits.keys())
    """
    discoverer = Discoverer()
    return discoverer.discover_schemas(system_description)

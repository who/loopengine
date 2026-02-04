"""AI genome discovery module using Claude API.

This module implements the discover_schemas function that sends system descriptions
to the Claude API and parses GenomeSchema objects from the response.

Also includes genome migration utilities for gracefully updating agent genomes
when schemas change (per PRD section 5.2).
"""

from __future__ import annotations

import json
import logging
import random
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


@dataclass
class MigrationResult:
    """Result of a genome migration operation.

    Tracks what changed during migration for debugging and auditing.

    Attributes:
        migrated_genome: The new genome dict after migration.
        added_traits: List of trait names that were added (new in schema).
        vestigial_traits: List of trait names that are now vestigial (not in new schema).
        preserved_traits: List of trait names that were preserved unchanged.
        schema_version: The version of the new schema.
    """

    migrated_genome: dict[str, float] = field(default_factory=dict)
    added_traits: list[str] = field(default_factory=list)
    vestigial_traits: list[str] = field(default_factory=list)
    preserved_traits: list[str] = field(default_factory=list)
    schema_version: int = 1


def migrate_genome(
    old_genome: dict[str, float],
    new_schema: GenomeSchema,
    vestigial_marker: str = "_vestigial",
) -> MigrationResult:
    """Migrate an agent genome to a new schema.

    Handles genome updates when schemas change per PRD section 5.2:
    - New traits: added with random initialization within the trait's range
    - Deprecated traits: preserved but flagged as vestigial
    - Existing traits: values preserved if trait name matches in new schema

    This function is non-destructive and does not modify the input genome.

    Args:
        old_genome: The agent's current genome dict (trait_name -> value).
        new_schema: The new GenomeSchema to migrate to.
        vestigial_marker: Suffix to append to deprecated trait names.

    Returns:
        MigrationResult with the migrated genome and change tracking.

    Example:
        >>> old_genome = {"speed": 0.8, "old_trait": 0.5}
        >>> new_schema = GenomeSchema(
        ...     role="worker",
        ...     traits={"speed": GenomeTrait(name="speed", description="..."),
        ...             "new_trait": GenomeTrait(name="new_trait", description="...")}
        ... )
        >>> result = migrate_genome(old_genome, new_schema)
        >>> "speed" in result.migrated_genome  # preserved
        True
        >>> "new_trait" in result.migrated_genome  # added
        True
        >>> "old_trait_vestigial" in result.migrated_genome  # marked vestigial
        True
    """
    result = MigrationResult(schema_version=new_schema.version)
    new_genome: dict[str, float] = {}

    # Get trait names from new schema
    new_trait_names = set(new_schema.traits.keys())
    old_trait_names = set(old_genome.keys())

    # Filter out already-vestigial traits from old_trait_names for comparison
    # (vestigial traits have the marker suffix)
    active_old_traits = {name for name in old_trait_names if not name.endswith(vestigial_marker)}
    vestigial_old_traits = {name for name in old_trait_names if name.endswith(vestigial_marker)}

    # 1. Preserve existing traits that are still in the schema
    for trait_name in active_old_traits & new_trait_names:
        new_genome[trait_name] = old_genome[trait_name]
        result.preserved_traits.append(trait_name)
        logger.debug("Preserved trait '%s' with value %.3f", trait_name, old_genome[trait_name])

    # 2. Add new traits with random initialization
    for trait_name in new_trait_names - active_old_traits:
        trait = new_schema.traits[trait_name]
        # Random value within the trait's range
        value = random.uniform(trait.min_val, trait.max_val)
        new_genome[trait_name] = value
        result.added_traits.append(trait_name)
        logger.debug(
            "Added new trait '%s' with random value %.3f (range: %.1f-%.1f)",
            trait_name,
            value,
            trait.min_val,
            trait.max_val,
        )

    # 3. Mark deprecated traits as vestigial
    for trait_name in active_old_traits - new_trait_names:
        vestigial_name = f"{trait_name}{vestigial_marker}"
        new_genome[vestigial_name] = old_genome[trait_name]
        result.vestigial_traits.append(trait_name)
        logger.debug(
            "Marked trait '%s' as vestigial (now '%s') with value %.3f",
            trait_name,
            vestigial_name,
            old_genome[trait_name],
        )

    # 4. Carry forward existing vestigial traits unchanged
    for trait_name in vestigial_old_traits:
        new_genome[trait_name] = old_genome[trait_name]
        logger.debug("Carried forward vestigial trait '%s'", trait_name)

    result.migrated_genome = new_genome

    # Log summary
    logger.info(
        "Genome migration complete: %d preserved, %d added, %d vestigial",
        len(result.preserved_traits),
        len(result.added_traits),
        len(result.vestigial_traits),
    )

    return result


# Valid trait categories
VALID_CATEGORIES = frozenset(["physical", "cognitive", "social", "temperamental", "skill"])

# Discovery prompt template based on PRD Appendix B
# This template is designed to produce consistent, role-appropriate genome schemas
# for simulation agents. Key design decisions:
# - Explicit category definitions with examples for consistent classification
# - Flexibility score guidance based on input predictability
# - Emphasis on role-specific traits (not generic personality traits)
# - Trait relevance to the role's actual responsibilities
DISCOVERY_PROMPT_TEMPLATE = """You are analyzing an organizational system to discover the meaningful
dimensions of variation for agents in each role. Your task is to identify traits that would
meaningfully affect an agent's performance in their specific role.

## SYSTEM DESCRIPTION
{system_description_json}

## INSTRUCTIONS

Analyze each role in the system and identify 3-6 traits per role that would meaningfully
affect performance. Traits should be:

1. **Role-specific**: Directly relevant to what this role does (not generic personality traits)
2. **Measurable**: Can vary on a 0.0-1.0 scale where higher is typically better (but not always)
3. **Impactful**: Would actually affect outcomes in the simulation
4. **Distinctive**: Different roles should have different trait profiles

## TRAIT CATEGORIES

Use ONLY these categories with the following meanings:

- **physical**: Traits related to speed, endurance, dexterity, stamina
  Examples: speed, endurance, dexterity, reaction_time

- **cognitive**: Traits related to thinking, planning, analysis, memory, learning
  Examples: analytical_ability, pattern_recognition, forecasting, attention_to_detail

- **social**: Traits related to interaction with others, communication, influence
  Examples: rapport_building, persuasion, conflict_resolution, empathy

- **temperamental**: Traits related to emotional regulation, stress response, disposition
  Examples: stress_tolerance, patience, adaptability, composure

- **skill**: Learned abilities specific to the role's domain
  Examples: technical_expertise, domain_knowledge, process_efficiency

## FLEXIBILITY SCORE

Assign a flexibility_score (0.0-1.0) based on how much input variance the role faces:

- **Low (0.1-0.3)**: Role has predictable, routine inputs. Same tasks day after day.
  Example: Assembly line worker, data entry clerk

- **Medium (0.4-0.6)**: Role has some variety but within expected patterns.
  Example: Cashier, sandwich maker (orders vary but ingredients/process known)

- **High (0.7-0.9)**: Role must handle unpredictable situations regularly.
  Example: Manager (unexpected problems), customer service (complaints), emergency responder

## TRAIT DESIGN GUIDELINES

For each role, consider:
- What inputs does this role receive? (from the role's inputs list)
- What outputs must this role produce? (from the role's outputs list)
- What constraints must this role operate within? (from the role's constraints)
- What links does this role have? (hierarchical = leadership traits, service = coordination)

Match traits to responsibilities:
- Roles handling customer interactions → social traits
- Roles doing physical work → physical and skill traits
- Roles making decisions → cognitive and temperamental traits
- Roles under time pressure → physical speed and temperamental stress tolerance

## OUTPUT FORMAT

Respond with ONLY valid JSON in this exact structure (no markdown, no commentary):

{{
    "roles": {{
        "role_name": {{
            "traits": [
                {{
                    "name": "trait_name_in_snake_case",
                    "description": "Clear description of what this trait represents",
                    "category": "one of: physical, cognitive, social, temperamental, skill",
                    "min_val": 0.0,
                    "max_val": 1.0
                }}
            ],
            "flexibility_score": 0.5
        }}
    }}
}}

Generate traits for ALL roles found in the system description."""


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
                # Validate and normalize category
                raw_category = trait_data.get("category", "").lower().strip()
                category = raw_category if raw_category in VALID_CATEGORIES else "skill"
                if raw_category and raw_category not in VALID_CATEGORIES:
                    logger.warning(
                        "Unknown category '%s' for trait '%s', defaulting to 'skill'",
                        raw_category,
                        trait_data.get("name", "unknown"),
                    )

                # Normalize trait name to snake_case
                raw_name = trait_data.get("name", "")
                name = raw_name.lower().replace(" ", "_").replace("-", "_")

                trait = GenomeTrait(
                    name=name,
                    description=trait_data.get("description", ""),
                    min_val=float(trait_data.get("min_val", 0.0)),
                    max_val=float(trait_data.get("max_val", 1.0)),
                    category=category,
                    discovered_at=discovered_at,
                )
                traits[trait.name] = trait

            # Validate and clamp flexibility score to 0.0-1.0
            raw_flexibility = float(role_data.get("flexibility_score", 0.5))
            flexibility = max(0.0, min(1.0, raw_flexibility))
            if raw_flexibility != flexibility:
                logger.warning(
                    "Flexibility score %s for role '%s' clamped to %s",
                    raw_flexibility,
                    role_name,
                    flexibility,
                )

            schema = GenomeSchema(
                role=role_name,
                traits=traits,
                discovered_at=discovered_at,
                discovery_prompt=discovery_prompt,
                version=1,
                flexibility_score=flexibility,
            )

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

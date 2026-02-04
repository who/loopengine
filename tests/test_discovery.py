"""Tests for the AI genome discovery module."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.discovery.discoverer import (
    DiscoveredRole,
    Discoverer,
    DiscoveryError,
    DiscoveryResult,
    discover_schemas,
)
from loopengine.model.genome import GenomeSchema, GenomeTrait


@pytest.fixture
def mock_config() -> LLMConfig:
    """Create a mock LLM config with test API key."""
    return LLMConfig(
        llm_provider=LLMProvider.CLAUDE,
        anthropic_api_key=SecretStr("test-api-key"),
        llm_max_tokens=500,
        llm_temperature=0.7,
        llm_timeout=30.0,
    )


@pytest.fixture
def sample_system_description() -> dict:
    """Create a sample sandwich shop system description."""
    return {
        "system": "Small sandwich shop, 3 employees",
        "roles": [
            {
                "name": "owner",
                "inputs": [
                    "supply_invoices",
                    "revenue_reports",
                    "staff_status",
                    "customer_flow_observation",
                ],
                "outputs": [
                    "supply_orders",
                    "directives",
                    "schedule_changes",
                    "pricing_decisions",
                ],
                "constraints": ["budget", "health_code", "operating_hours"],
                "links_to": ["sandwich_maker (hierarchical)", "cashier (hierarchical)"],
            },
            {
                "name": "sandwich_maker",
                "inputs": ["order_tickets", "ingredients", "directives"],
                "outputs": ["finished_sandwiches", "status_reports", "waste"],
                "constraints": ["food_safety", "recipe_standards"],
                "links_to": ["owner (hierarchical)", "cashier (service)"],
            },
            {
                "name": "cashier",
                "inputs": ["customers", "finished_sandwiches", "directives"],
                "outputs": ["order_tickets", "served_customers", "revenue_reports"],
                "constraints": ["cash_handling", "customer_service"],
                "links_to": ["owner (hierarchical)", "sandwich_maker (service)"],
            },
        ],
    }


@pytest.fixture
def mock_discovery_response() -> str:
    """Create a mock discovery response from Claude API."""
    return json.dumps(
        {
            "roles": {
                "owner": {
                    "traits": [
                        {
                            "name": "supply_forecasting",
                            "description": "Ability to predict supply needs",
                            "category": "cognitive",
                            "min_val": 0.0,
                            "max_val": 1.0,
                        },
                        {
                            "name": "delegation",
                            "description": "Effectiveness at delegating tasks",
                            "category": "social",
                            "min_val": 0.0,
                            "max_val": 1.0,
                        },
                        {
                            "name": "cost_sensitivity",
                            "description": "Focus on cost efficiency",
                            "category": "temperamental",
                            "min_val": 0.0,
                            "max_val": 1.0,
                        },
                    ],
                    "flexibility_score": 0.6,
                },
                "sandwich_maker": {
                    "traits": [
                        {
                            "name": "speed",
                            "description": "Sandwich assembly speed",
                            "category": "physical",
                            "min_val": 0.0,
                            "max_val": 1.0,
                        },
                        {
                            "name": "consistency",
                            "description": "Product consistency",
                            "category": "skill",
                            "min_val": 0.0,
                            "max_val": 1.0,
                        },
                        {
                            "name": "stress_tolerance",
                            "description": "Ability to handle pressure",
                            "category": "temperamental",
                            "min_val": 0.0,
                            "max_val": 1.0,
                        },
                    ],
                    "flexibility_score": 0.3,
                },
                "cashier": {
                    "traits": [
                        {
                            "name": "customer_rapport",
                            "description": "Ability to connect with customers",
                            "category": "social",
                            "min_val": 0.0,
                            "max_val": 1.0,
                        },
                        {
                            "name": "order_accuracy",
                            "description": "Accuracy in taking orders",
                            "category": "cognitive",
                            "min_val": 0.0,
                            "max_val": 1.0,
                        },
                    ],
                    "flexibility_score": 0.4,
                },
            }
        }
    )


@pytest.fixture
def mock_anthropic_response(mock_discovery_response: str) -> MagicMock:
    """Create a mock Anthropic API response."""
    response = MagicMock()
    response.content = [MagicMock(type="text", text=mock_discovery_response)]
    response.model = "claude-sonnet-4-20250514"
    response.usage = MagicMock(input_tokens=500, output_tokens=300)
    response.stop_reason = "end_turn"
    return response


class TestDiscovererInit:
    """Tests for Discoverer initialization."""

    def test_init_with_config(self, mock_config: LLMConfig) -> None:
        """Test discoverer initializes with provided config."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            discoverer = Discoverer(config=mock_config)
            assert discoverer._config == mock_config
            mock_anthropic.assert_called_once()

    def test_init_without_api_key_logs_warning(self) -> None:
        """Test discoverer logs warning when no API key is set."""
        config = LLMConfig(
            llm_provider=LLMProvider.CLAUDE,
            anthropic_api_key=None,
        )
        with patch("loopengine.discovery.discoverer.logger") as mock_logger:
            discoverer = Discoverer(config=config)
            mock_logger.warning.assert_called_once()
            assert discoverer._client is None

    def test_init_sets_longer_timeout(self, mock_config: LLMConfig) -> None:
        """Test discoverer uses longer timeout for discovery."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            Discoverer(config=mock_config)
            call_kwargs = mock_anthropic.call_args.kwargs
            # Should use 2x the configured timeout
            assert call_kwargs["timeout"] == mock_config.llm_timeout * 2


class TestDiscoverSchemas:
    """Tests for discover_schemas method."""

    def test_discover_schemas_success(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test successful schema discovery returns GenomeSchemas."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            assert len(schemas) == 3
            assert "owner" in schemas
            assert "sandwich_maker" in schemas
            assert "cashier" in schemas

            # Verify schema structure
            owner_schema = schemas["owner"]
            assert isinstance(owner_schema, GenomeSchema)
            assert owner_schema.role == "owner"
            assert len(owner_schema.traits) == 3
            assert "supply_forecasting" in owner_schema.traits
            assert "delegation" in owner_schema.traits
            assert "cost_sensitivity" in owner_schema.traits

    def test_discover_schemas_returns_correct_trait_details(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test discovered traits have correct details."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            speed_trait = schemas["sandwich_maker"].traits["speed"]
            assert isinstance(speed_trait, GenomeTrait)
            assert speed_trait.name == "speed"
            assert speed_trait.description == "Sandwich assembly speed"
            assert speed_trait.category == "physical"
            assert speed_trait.min_val == 0.0
            assert speed_trait.max_val == 1.0
            assert isinstance(speed_trait.discovered_at, datetime)

    def test_discover_schemas_without_api_key_raises_error(
        self, sample_system_description: dict
    ) -> None:
        """Test discovery raises error when API key not set."""
        config = LLMConfig(
            llm_provider=LLMProvider.CLAUDE,
            anthropic_api_key=None,
        )
        discoverer = Discoverer(config=config)

        with pytest.raises(DiscoveryError) as exc_info:
            discoverer.discover_schemas(sample_system_description)

        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_discover_schemas_handles_api_timeout(
        self, mock_config: LLMConfig, sample_system_description: dict
    ) -> None:
        """Test discovery handles timeout errors with retry."""
        from anthropic import APITimeoutError

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            # Fail all retries
            mock_client.messages.create.side_effect = APITimeoutError(request=MagicMock())
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)

            with pytest.raises(DiscoveryError) as exc_info:
                discoverer.discover_schemas(sample_system_description)

            assert "API error" in str(exc_info.value) or "timeout" in str(exc_info.value).lower()

    def test_discover_schemas_handles_rate_limit(
        self, mock_config: LLMConfig, sample_system_description: dict
    ) -> None:
        """Test discovery handles rate limit errors with retry."""
        from anthropic import RateLimitError

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)

            with pytest.raises(DiscoveryError) as exc_info:
                discoverer.discover_schemas(sample_system_description)

            assert "API error" in str(exc_info.value) or "Rate limit" in str(exc_info.value)

    def test_discover_schemas_handles_api_error(
        self, mock_config: LLMConfig, sample_system_description: dict
    ) -> None:
        """Test discovery handles generic API errors."""
        from anthropic import APIError

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = APIError(
                message="Internal server error",
                request=MagicMock(),
                body=None,
            )
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)

            with pytest.raises(DiscoveryError) as exc_info:
                discoverer.discover_schemas(sample_system_description)

            assert "API error" in str(exc_info.value)


class TestDiscoverFull:
    """Tests for discover method with full metadata."""

    def test_discover_returns_discovery_result(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test discover returns DiscoveryResult with metadata."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            result = discoverer.discover(sample_system_description)

            assert isinstance(result, DiscoveryResult)
            assert len(result.roles) == 3
            assert result.discovery_prompt != ""
            assert isinstance(result.discovered_at, datetime)

    def test_discover_includes_flexibility_scores(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test discover includes flexibility scores per role."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            result = discoverer.discover(sample_system_description)

            assert result.roles["owner"].flexibility_score == 0.6
            assert result.roles["sandwich_maker"].flexibility_score == 0.3
            assert result.roles["cashier"].flexibility_score == 0.4


class TestResponseParsing:
    """Tests for response parsing logic."""

    def test_parse_json_with_code_block(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_discovery_response: str,
    ) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        response = MagicMock()
        response.content = [MagicMock(type="text", text=f"```json\n{mock_discovery_response}\n```")]
        response.model = "claude-sonnet-4-20250514"
        response.usage = MagicMock(input_tokens=500, output_tokens=300)
        response.stop_reason = "end_turn"

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            assert len(schemas) == 3
            assert "owner" in schemas

    def test_parse_invalid_json_raises_error(
        self, mock_config: LLMConfig, sample_system_description: dict
    ) -> None:
        """Test invalid JSON response raises DiscoveryError."""
        response = MagicMock()
        response.content = [MagicMock(type="text", text="This is not valid JSON")]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)

            with pytest.raises(DiscoveryError) as exc_info:
                discoverer.discover_schemas(sample_system_description)

            assert "JSON" in str(exc_info.value)

    def test_parse_missing_roles_key_raises_error(
        self, mock_config: LLMConfig, sample_system_description: dict
    ) -> None:
        """Test response missing roles key raises DiscoveryError."""
        response = MagicMock()
        response.content = [MagicMock(type="text", text='{"data": "no roles"}')]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)

            with pytest.raises(DiscoveryError) as exc_info:
                discoverer.discover_schemas(sample_system_description)

            assert "roles" in str(exc_info.value).lower()

    def test_parse_empty_response_raises_error(
        self, mock_config: LLMConfig, sample_system_description: dict
    ) -> None:
        """Test empty response raises DiscoveryError."""
        response = MagicMock()
        response.content = []

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)

            with pytest.raises(DiscoveryError) as exc_info:
                discoverer.discover_schemas(sample_system_description)

            assert "Empty response" in str(exc_info.value)

    def test_parse_handles_missing_optional_fields(
        self, mock_config: LLMConfig, sample_system_description: dict
    ) -> None:
        """Test parser handles missing optional fields gracefully."""
        minimal_response = json.dumps(
            {
                "roles": {
                    "owner": {
                        "traits": [
                            {
                                "name": "leadership",
                                "description": "Leadership ability",
                            }
                        ],
                    }
                }
            }
        )
        response = MagicMock()
        response.content = [MagicMock(type="text", text=minimal_response)]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            # Should use defaults for missing fields
            # Empty/missing category defaults to "skill"
            trait = schemas["owner"].traits["leadership"]
            assert trait.min_val == 0.0
            assert trait.max_val == 1.0
            assert trait.category == "skill"  # Empty category defaults to "skill"


class TestConvenienceFunction:
    """Tests for the discover_schemas convenience function."""

    def test_discover_schemas_function(
        self,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test the convenience function works correctly."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            with patch("loopengine.discovery.discoverer.get_llm_config") as mock_get_config:
                mock_get_config.return_value = LLMConfig(
                    llm_provider=LLMProvider.CLAUDE,
                    anthropic_api_key=SecretStr("test-key"),
                )

                schemas = discover_schemas(sample_system_description)

                assert len(schemas) == 3
                assert "owner" in schemas
                assert "sandwich_maker" in schemas
                assert "cashier" in schemas


class TestDiscoveredRole:
    """Tests for DiscoveredRole dataclass."""

    def test_discovered_role_defaults(self) -> None:
        """Test DiscoveredRole has correct defaults."""
        schema = GenomeSchema(role="test")
        role = DiscoveredRole(schema=schema)

        assert role.schema == schema
        assert role.flexibility_score == 0.5

    def test_discovered_role_custom_flexibility(self) -> None:
        """Test DiscoveredRole accepts custom flexibility."""
        schema = GenomeSchema(role="test")
        role = DiscoveredRole(schema=schema, flexibility_score=0.8)

        assert role.flexibility_score == 0.8


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_discovery_result_defaults(self) -> None:
        """Test DiscoveryResult has correct defaults."""
        result = DiscoveryResult()

        assert result.roles == {}
        assert result.discovery_prompt == ""
        assert isinstance(result.discovered_at, datetime)

    def test_discovery_result_with_data(self) -> None:
        """Test DiscoveryResult accepts data."""
        schema = GenomeSchema(role="test")
        role = DiscoveredRole(schema=schema, flexibility_score=0.7)

        result = DiscoveryResult(
            roles={"test": role},
            discovery_prompt="test prompt",
        )

        assert "test" in result.roles
        assert result.discovery_prompt == "test prompt"


class TestModuleExports:
    """Tests for module exports."""

    def test_import_from_discovery_package(self) -> None:
        """Test all exports are available from discovery package."""
        from loopengine.discovery import (
            DiscoveredRole,
            Discoverer,
            DiscoveryError,
            DiscoveryResult,
            discover_schemas,
        )

        assert DiscoveredRole is not None
        assert DiscoveryError is not None
        assert DiscoveryResult is not None
        assert Discoverer is not None
        assert discover_schemas is not None


class TestPromptConstruction:
    """Tests for prompt construction."""

    def test_prompt_includes_system_description(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test the prompt includes the system description."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            discoverer.discover_schemas(sample_system_description)

            # Get the call arguments
            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            prompt_content = messages[0]["content"]

            # Verify system description is in the prompt
            assert "sandwich shop" in prompt_content.lower()
            assert "owner" in prompt_content
            assert "sandwich_maker" in prompt_content
            assert "cashier" in prompt_content

    def test_prompt_uses_low_temperature(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test discovery uses low temperature for structured output."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            discoverer.discover_schemas(sample_system_description)

            call_args = mock_client.messages.create.call_args
            temperature = call_args.kwargs["temperature"]

            # Should use low temperature for structured output
            assert temperature <= 0.5

    def test_prompt_includes_category_definitions(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test the prompt includes explicit category definitions."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            discoverer.discover_schemas(sample_system_description)

            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            prompt_content = messages[0]["content"]

            # Verify all categories are defined
            assert "physical" in prompt_content.lower()
            assert "cognitive" in prompt_content.lower()
            assert "social" in prompt_content.lower()
            assert "temperamental" in prompt_content.lower()
            assert "skill" in prompt_content.lower()

    def test_prompt_includes_flexibility_guidance(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test the prompt includes flexibility score guidance."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            discoverer.discover_schemas(sample_system_description)

            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            prompt_content = messages[0]["content"]

            # Verify flexibility guidance is in the prompt
            assert "flexibility_score" in prompt_content
            assert "0.0" in prompt_content or "0.1" in prompt_content
            assert "1.0" in prompt_content or "0.9" in prompt_content


class TestPromptTemplateDesign:
    """Tests for prompt template design and structure."""

    def test_prompt_template_contains_role_specific_guidance(self) -> None:
        """Test that prompt template guides toward role-specific traits."""
        from loopengine.discovery.discoverer import DISCOVERY_PROMPT_TEMPLATE

        prompt = DISCOVERY_PROMPT_TEMPLATE.lower()
        assert "role-specific" in prompt or "role specific" in prompt
        assert "inputs" in prompt
        assert "outputs" in prompt
        assert "links" in prompt

    def test_prompt_template_contains_json_format(self) -> None:
        """Test that prompt template specifies JSON output format."""
        from loopengine.discovery.discoverer import DISCOVERY_PROMPT_TEMPLATE

        # Should contain JSON structure example
        assert '"roles"' in DISCOVERY_PROMPT_TEMPLATE
        assert '"traits"' in DISCOVERY_PROMPT_TEMPLATE
        assert '"name"' in DISCOVERY_PROMPT_TEMPLATE
        assert '"description"' in DISCOVERY_PROMPT_TEMPLATE
        assert '"category"' in DISCOVERY_PROMPT_TEMPLATE

    def test_prompt_template_contains_valid_categories(self) -> None:
        """Test that prompt template lists valid categories."""
        from loopengine.discovery.discoverer import (
            DISCOVERY_PROMPT_TEMPLATE,
            VALID_CATEGORIES,
        )

        for category in VALID_CATEGORIES:
            assert category in DISCOVERY_PROMPT_TEMPLATE.lower()


class TestCategoryValidation:
    """Tests for category validation in parsing."""

    def test_valid_categories_constant(self) -> None:
        """Test VALID_CATEGORIES contains expected values."""
        from loopengine.discovery.discoverer import VALID_CATEGORIES

        assert "physical" in VALID_CATEGORIES
        assert "cognitive" in VALID_CATEGORIES
        assert "social" in VALID_CATEGORIES
        assert "temperamental" in VALID_CATEGORIES
        assert "skill" in VALID_CATEGORIES
        assert len(VALID_CATEGORIES) == 5

    def test_import_valid_categories_from_package(self) -> None:
        """Test VALID_CATEGORIES can be imported from discovery package."""
        from loopengine.discovery import VALID_CATEGORIES

        assert isinstance(VALID_CATEGORIES, frozenset)
        assert len(VALID_CATEGORIES) == 5

    def test_invalid_category_defaults_to_skill(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
    ) -> None:
        """Test that invalid categories default to 'skill'."""
        invalid_response = json.dumps(
            {
                "roles": {
                    "owner": {
                        "traits": [
                            {
                                "name": "leadership",
                                "description": "Leadership ability",
                                "category": "invalid_category",
                                "min_val": 0.0,
                                "max_val": 1.0,
                            }
                        ],
                        "flexibility_score": 0.5,
                    }
                }
            }
        )
        response = MagicMock()
        response.content = [MagicMock(type="text", text=invalid_response)]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            # Invalid category should default to skill
            trait = schemas["owner"].traits["leadership"]
            assert trait.category == "skill"

    def test_category_case_normalization(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
    ) -> None:
        """Test that category names are normalized to lowercase."""
        mixed_case_response = json.dumps(
            {
                "roles": {
                    "owner": {
                        "traits": [
                            {
                                "name": "planning",
                                "description": "Planning ability",
                                "category": "COGNITIVE",
                                "min_val": 0.0,
                                "max_val": 1.0,
                            },
                            {
                                "name": "endurance",
                                "description": "Physical endurance",
                                "category": "Physical",
                                "min_val": 0.0,
                                "max_val": 1.0,
                            },
                        ],
                        "flexibility_score": 0.5,
                    }
                }
            }
        )
        response = MagicMock()
        response.content = [MagicMock(type="text", text=mixed_case_response)]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            assert schemas["owner"].traits["planning"].category == "cognitive"
            assert schemas["owner"].traits["endurance"].category == "physical"


class TestTraitNameNormalization:
    """Tests for trait name normalization."""

    def test_trait_name_normalized_to_snake_case(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
    ) -> None:
        """Test that trait names are normalized to snake_case."""
        response_with_spaces = json.dumps(
            {
                "roles": {
                    "owner": {
                        "traits": [
                            {
                                "name": "Problem Solving",
                                "description": "Problem solving ability",
                                "category": "cognitive",
                            },
                            {
                                "name": "stress-tolerance",
                                "description": "Stress tolerance",
                                "category": "temperamental",
                            },
                        ],
                        "flexibility_score": 0.5,
                    }
                }
            }
        )
        response = MagicMock()
        response.content = [MagicMock(type="text", text=response_with_spaces)]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            # Names should be normalized
            assert "problem_solving" in schemas["owner"].traits
            assert "stress_tolerance" in schemas["owner"].traits


class TestFlexibilityScoreValidation:
    """Tests for flexibility score validation."""

    def test_flexibility_score_clamped_to_valid_range(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
    ) -> None:
        """Test that flexibility scores outside 0-1 are clamped."""
        response_with_invalid_flexibility = json.dumps(
            {
                "roles": {
                    "role_low": {
                        "traits": [
                            {"name": "trait1", "description": "Trait 1", "category": "skill"}
                        ],
                        "flexibility_score": -0.5,
                    },
                    "role_high": {
                        "traits": [
                            {"name": "trait2", "description": "Trait 2", "category": "skill"}
                        ],
                        "flexibility_score": 1.5,
                    },
                    "role_valid": {
                        "traits": [
                            {"name": "trait3", "description": "Trait 3", "category": "skill"}
                        ],
                        "flexibility_score": 0.7,
                    },
                }
            }
        )
        response = MagicMock()
        response.content = [MagicMock(type="text", text=response_with_invalid_flexibility)]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            result = discoverer.discover(sample_system_description)

            # Scores should be clamped
            assert result.roles["role_low"].flexibility_score == 0.0
            assert result.roles["role_high"].flexibility_score == 1.0
            assert result.roles["role_valid"].flexibility_score == 0.7


class TestVariousSystemDescriptions:
    """Tests for handling various system descriptions."""

    @pytest.fixture
    def software_team_description(self) -> dict:
        """Create a software team system description."""
        return {
            "system": "Software development team with 4 members",
            "roles": [
                {
                    "name": "product_manager",
                    "inputs": ["feature_requests", "bug_reports", "stakeholder_feedback"],
                    "outputs": ["prioritized_backlog", "requirements", "decisions"],
                    "constraints": ["timeline", "budget", "scope"],
                    "links_to": ["developer (hierarchical)", "designer (hierarchical)"],
                },
                {
                    "name": "developer",
                    "inputs": ["requirements", "code_reviews", "bug_tickets"],
                    "outputs": ["code", "pull_requests", "documentation"],
                    "constraints": ["coding_standards", "security_guidelines"],
                    "links_to": [
                        "product_manager (hierarchical)",
                        "developer (peer)",
                        "designer (service)",
                    ],
                },
                {
                    "name": "designer",
                    "inputs": ["requirements", "user_research", "feedback"],
                    "outputs": ["designs", "prototypes", "style_guides"],
                    "constraints": ["brand_guidelines", "accessibility"],
                    "links_to": ["product_manager (hierarchical)", "developer (service)"],
                },
            ],
        }

    @pytest.fixture
    def mock_software_team_response(self) -> str:
        """Create a mock discovery response for software team."""
        return json.dumps(
            {
                "roles": {
                    "product_manager": {
                        "traits": [
                            {
                                "name": "prioritization",
                                "description": "Ability to prioritize tasks effectively",
                                "category": "cognitive",
                                "min_val": 0.0,
                                "max_val": 1.0,
                            },
                            {
                                "name": "stakeholder_communication",
                                "description": "Communication with stakeholders",
                                "category": "social",
                                "min_val": 0.0,
                                "max_val": 1.0,
                            },
                        ],
                        "flexibility_score": 0.7,
                    },
                    "developer": {
                        "traits": [
                            {
                                "name": "coding_speed",
                                "description": "Speed of writing code",
                                "category": "skill",
                                "min_val": 0.0,
                                "max_val": 1.0,
                            },
                            {
                                "name": "debugging",
                                "description": "Debugging ability",
                                "category": "cognitive",
                                "min_val": 0.0,
                                "max_val": 1.0,
                            },
                        ],
                        "flexibility_score": 0.5,
                    },
                    "designer": {
                        "traits": [
                            {
                                "name": "creativity",
                                "description": "Creative design ability",
                                "category": "cognitive",
                                "min_val": 0.0,
                                "max_val": 1.0,
                            },
                            {
                                "name": "user_empathy",
                                "description": "Understanding of user needs",
                                "category": "social",
                                "min_val": 0.0,
                                "max_val": 1.0,
                            },
                        ],
                        "flexibility_score": 0.6,
                    },
                }
            }
        )

    def test_discovery_with_different_system_description(
        self,
        mock_config: LLMConfig,
        software_team_description: dict,
        mock_software_team_response: str,
    ) -> None:
        """Test discovery works with different system descriptions."""
        response = MagicMock()
        response.content = [MagicMock(type="text", text=mock_software_team_response)]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(software_team_description)

            assert len(schemas) == 3
            assert "product_manager" in schemas
            assert "developer" in schemas
            assert "designer" in schemas

    def test_prompt_includes_software_team_roles(
        self,
        mock_config: LLMConfig,
        software_team_description: dict,
        mock_software_team_response: str,
    ) -> None:
        """Test prompt includes software team role information."""
        response = MagicMock()
        response.content = [MagicMock(type="text", text=mock_software_team_response)]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            discoverer.discover_schemas(software_team_description)

            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            prompt_content = messages[0]["content"]

            assert "product_manager" in prompt_content
            assert "developer" in prompt_content
            assert "designer" in prompt_content
            assert "feature_requests" in prompt_content

    def test_flexibility_appropriate_for_role(
        self,
        mock_config: LLMConfig,
        software_team_description: dict,
        mock_software_team_response: str,
    ) -> None:
        """Test flexibility scores are appropriate for roles."""
        response = MagicMock()
        response.content = [MagicMock(type="text", text=mock_software_team_response)]

        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            result = discoverer.discover(software_team_description)

            # PM deals with more unpredictable inputs than developer
            assert (
                result.roles["product_manager"].flexibility_score
                >= result.roles["developer"].flexibility_score
            )


class TestRoleAppropriateTraits:
    """Tests for role-appropriate trait discovery."""

    def test_sandwich_maker_gets_physical_traits(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test sandwich maker role gets physical traits like speed."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            # Sandwich maker should have physical trait
            maker_traits = schemas["sandwich_maker"].traits
            categories = [t.category for t in maker_traits.values()]
            assert "physical" in categories or "skill" in categories

    def test_cashier_gets_social_traits(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test cashier role gets social traits for customer interaction."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            # Cashier should have social trait
            cashier_traits = schemas["cashier"].traits
            categories = [t.category for t in cashier_traits.values()]
            assert "social" in categories

    def test_owner_gets_cognitive_traits(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test owner role gets cognitive traits for decision making."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            schemas = discoverer.discover_schemas(sample_system_description)

            # Owner should have cognitive traits for management
            owner_traits = schemas["owner"].traits
            categories = [t.category for t in owner_traits.values()]
            assert "cognitive" in categories or "social" in categories

    def test_flexibility_sandwich_maker_less_than_owner(
        self,
        mock_config: LLMConfig,
        sample_system_description: dict,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Test sandwich maker has lower flexibility than owner (more predictable work)."""
        with patch("loopengine.discovery.discoverer.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            discoverer = Discoverer(config=mock_config)
            result = discoverer.discover(sample_system_description)

            # Sandwich maker should have lower flexibility than owner
            # (sandwich making is more routine than managing)
            assert (
                result.roles["sandwich_maker"].flexibility_score
                <= result.roles["owner"].flexibility_score
            )


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_migration_result_defaults(self) -> None:
        """Test MigrationResult has correct defaults."""
        from loopengine.discovery.discoverer import MigrationResult

        result = MigrationResult()

        assert result.migrated_genome == {}
        assert result.added_traits == []
        assert result.vestigial_traits == []
        assert result.preserved_traits == []
        assert result.schema_version == 1

    def test_migration_result_with_data(self) -> None:
        """Test MigrationResult accepts data."""
        from loopengine.discovery.discoverer import MigrationResult

        result = MigrationResult(
            migrated_genome={"speed": 0.5},
            added_traits=["new_trait"],
            vestigial_traits=["old_trait"],
            preserved_traits=["speed"],
            schema_version=2,
        )

        assert result.migrated_genome == {"speed": 0.5}
        assert result.added_traits == ["new_trait"]
        assert result.vestigial_traits == ["old_trait"]
        assert result.preserved_traits == ["speed"]
        assert result.schema_version == 2


class TestMigrateGenome:
    """Tests for migrate_genome function."""

    @pytest.fixture
    def old_schema(self) -> GenomeSchema:
        """Create an old schema for testing."""
        return GenomeSchema(
            role="worker",
            traits={
                "speed": GenomeTrait(
                    name="speed", description="Movement speed", min_val=0.0, max_val=1.0
                ),
                "strength": GenomeTrait(
                    name="strength", description="Physical strength", min_val=0.0, max_val=1.0
                ),
                "old_trait": GenomeTrait(
                    name="old_trait", description="Will be deprecated", min_val=0.0, max_val=1.0
                ),
            },
            version=1,
        )

    @pytest.fixture
    def new_schema(self) -> GenomeSchema:
        """Create a new schema for testing migration."""
        return GenomeSchema(
            role="worker",
            traits={
                "speed": GenomeTrait(
                    name="speed", description="Movement speed", min_val=0.0, max_val=1.0
                ),
                "strength": GenomeTrait(
                    name="strength", description="Physical strength", min_val=0.0, max_val=1.0
                ),
                "new_trait": GenomeTrait(
                    name="new_trait", description="Newly discovered", min_val=0.2, max_val=0.8
                ),
            },
            version=2,
        )

    @pytest.fixture
    def old_genome(self) -> dict[str, float]:
        """Create an old genome for testing."""
        return {"speed": 0.7, "strength": 0.5, "old_trait": 0.3}

    def test_migrate_preserves_existing_traits(
        self, old_genome: dict[str, float], new_schema: GenomeSchema
    ) -> None:
        """Test that existing traits matching the new schema are preserved."""
        from loopengine.discovery.discoverer import migrate_genome

        result = migrate_genome(old_genome, new_schema)

        # speed and strength should be preserved with exact values
        assert result.migrated_genome["speed"] == 0.7
        assert result.migrated_genome["strength"] == 0.5
        assert "speed" in result.preserved_traits
        assert "strength" in result.preserved_traits

    def test_migrate_adds_new_traits_with_random_values(
        self, old_genome: dict[str, float], new_schema: GenomeSchema
    ) -> None:
        """Test that new traits are added with random values in range."""
        from loopengine.discovery.discoverer import migrate_genome

        result = migrate_genome(old_genome, new_schema)

        # new_trait should be added
        assert "new_trait" in result.migrated_genome
        assert "new_trait" in result.added_traits

        # Value should be within the trait's range (0.2-0.8)
        new_value = result.migrated_genome["new_trait"]
        assert 0.2 <= new_value <= 0.8

    def test_migrate_marks_deprecated_traits_vestigial(
        self, old_genome: dict[str, float], new_schema: GenomeSchema
    ) -> None:
        """Test that deprecated traits are marked as vestigial."""
        from loopengine.discovery.discoverer import migrate_genome

        result = migrate_genome(old_genome, new_schema)

        # old_trait should be renamed with vestigial marker
        assert "old_trait" not in result.migrated_genome
        assert "old_trait_vestigial" in result.migrated_genome
        assert result.migrated_genome["old_trait_vestigial"] == 0.3
        assert "old_trait" in result.vestigial_traits

    def test_migrate_is_non_destructive(
        self, old_genome: dict[str, float], new_schema: GenomeSchema
    ) -> None:
        """Test that migration does not modify the input genome."""
        from loopengine.discovery.discoverer import migrate_genome

        original_genome = old_genome.copy()
        migrate_genome(old_genome, new_schema)

        # Original genome should be unchanged
        assert old_genome == original_genome

    def test_migrate_returns_schema_version(
        self, old_genome: dict[str, float], new_schema: GenomeSchema
    ) -> None:
        """Test that migration result includes schema version."""
        from loopengine.discovery.discoverer import migrate_genome

        result = migrate_genome(old_genome, new_schema)

        assert result.schema_version == new_schema.version

    def test_migrate_custom_vestigial_marker(
        self, old_genome: dict[str, float], new_schema: GenomeSchema
    ) -> None:
        """Test that custom vestigial marker can be used."""
        from loopengine.discovery.discoverer import migrate_genome

        result = migrate_genome(old_genome, new_schema, vestigial_marker="_deprecated")

        assert "old_trait_deprecated" in result.migrated_genome
        assert "old_trait_vestigial" not in result.migrated_genome

    def test_migrate_empty_genome_to_new_schema(self, new_schema: GenomeSchema) -> None:
        """Test migrating an empty genome adds all traits."""
        from loopengine.discovery.discoverer import migrate_genome

        result = migrate_genome({}, new_schema)

        assert len(result.migrated_genome) == 3
        assert "speed" in result.migrated_genome
        assert "strength" in result.migrated_genome
        assert "new_trait" in result.migrated_genome
        assert len(result.added_traits) == 3
        assert len(result.preserved_traits) == 0
        assert len(result.vestigial_traits) == 0

    def test_migrate_to_empty_schema(self, old_genome: dict[str, float]) -> None:
        """Test migrating to an empty schema marks all traits vestigial."""
        from loopengine.discovery.discoverer import migrate_genome

        empty_schema = GenomeSchema(role="worker", traits={}, version=3)
        result = migrate_genome(old_genome, empty_schema)

        # All traits should be vestigial
        assert len(result.vestigial_traits) == 3
        assert "speed" in result.vestigial_traits
        assert "strength" in result.vestigial_traits
        assert "old_trait" in result.vestigial_traits

        # Vestigial traits should have values preserved
        assert result.migrated_genome["speed_vestigial"] == 0.7
        assert result.migrated_genome["strength_vestigial"] == 0.5
        assert result.migrated_genome["old_trait_vestigial"] == 0.3

    def test_migrate_preserves_existing_vestigial_traits(self) -> None:
        """Test that already-vestigial traits are carried forward."""
        from loopengine.discovery.discoverer import migrate_genome

        # Genome with a vestigial trait from a previous migration
        genome_with_vestigial = {
            "speed": 0.7,
            "ancient_trait_vestigial": 0.2,  # Already vestigial
        }

        schema = GenomeSchema(
            role="worker",
            traits={
                "speed": GenomeTrait(name="speed", description="Speed"),
                "new_skill": GenomeTrait(name="new_skill", description="New"),
            },
            version=4,
        )

        result = migrate_genome(genome_with_vestigial, schema)

        # Ancient vestigial trait should still be present
        assert "ancient_trait_vestigial" in result.migrated_genome
        assert result.migrated_genome["ancient_trait_vestigial"] == 0.2

        # speed should be preserved
        assert result.migrated_genome["speed"] == 0.7

        # new_skill should be added
        assert "new_skill" in result.migrated_genome

    def test_migrate_logs_changes(
        self, old_genome: dict[str, float], new_schema: GenomeSchema
    ) -> None:
        """Test that migration logs changes for debugging."""
        from loopengine.discovery.discoverer import migrate_genome

        with patch("loopengine.discovery.discoverer.logger") as mock_logger:
            migrate_genome(old_genome, new_schema)

            # Should log preserved traits
            mock_logger.debug.assert_any_call("Preserved trait '%s' with value %.3f", "speed", 0.7)
            mock_logger.debug.assert_any_call(
                "Preserved trait '%s' with value %.3f", "strength", 0.5
            )

            # Should log vestigial traits
            mock_logger.debug.assert_any_call(
                "Marked trait '%s' as vestigial (now '%s') with value %.3f",
                "old_trait",
                "old_trait_vestigial",
                0.3,
            )

            # Should log summary
            mock_logger.info.assert_called_with(
                "Genome migration complete: %d preserved, %d added, %d vestigial",
                2,  # preserved: speed, strength
                1,  # added: new_trait
                1,  # vestigial: old_trait
            )

    def test_migrate_new_trait_values_within_range(self) -> None:
        """Test that new trait values are always within the specified range."""
        from loopengine.discovery.discoverer import migrate_genome

        schema = GenomeSchema(
            role="test",
            traits={
                "trait_narrow": GenomeTrait(
                    name="trait_narrow", description="Narrow range", min_val=0.4, max_val=0.6
                ),
                "trait_wide": GenomeTrait(
                    name="trait_wide", description="Wide range", min_val=0.0, max_val=1.0
                ),
            },
            version=1,
        )

        # Run multiple times to test randomness
        for _ in range(20):
            result = migrate_genome({}, schema)

            narrow_val = result.migrated_genome["trait_narrow"]
            wide_val = result.migrated_genome["trait_wide"]

            assert 0.4 <= narrow_val <= 0.6, f"trait_narrow value {narrow_val} out of range"
            assert 0.0 <= wide_val <= 1.0, f"trait_wide value {wide_val} out of range"

    def test_migrate_same_schema_no_changes(self) -> None:
        """Test migrating to the same schema preserves all traits."""
        from loopengine.discovery.discoverer import migrate_genome

        genome = {"speed": 0.7, "strength": 0.5}
        schema = GenomeSchema(
            role="worker",
            traits={
                "speed": GenomeTrait(name="speed", description="Speed"),
                "strength": GenomeTrait(name="strength", description="Strength"),
            },
            version=1,
        )

        result = migrate_genome(genome, schema)

        assert result.migrated_genome == genome
        assert len(result.preserved_traits) == 2
        assert len(result.added_traits) == 0
        assert len(result.vestigial_traits) == 0


class TestMigrationWithAgents:
    """Tests for genome migration with agents (integration scenarios)."""

    def test_migrate_agent_genome_preserves_functionality(self) -> None:
        """Test that migrated genome works with agent policy expectations."""
        from loopengine.discovery.discoverer import migrate_genome
        from loopengine.model.agent import Agent

        # Create an agent with an old genome
        agent = Agent(
            id="agent-1",
            name="Test Worker",
            role="worker",
            genome={"speed": 0.8, "old_skill": 0.6},
        )

        # New schema without old_skill but with new_skill
        new_schema = GenomeSchema(
            role="worker",
            traits={
                "speed": GenomeTrait(name="speed", description="Speed"),
                "new_skill": GenomeTrait(name="new_skill", description="New skill"),
            },
            version=2,
        )

        # Migrate the genome
        result = migrate_genome(agent.genome, new_schema)

        # Update agent genome
        agent.genome = result.migrated_genome

        # Agent should still have speed
        assert "speed" in agent.genome
        assert agent.genome["speed"] == 0.8

        # Agent should have new_skill
        assert "new_skill" in agent.genome

        # Old skill should be vestigial
        assert "old_skill_vestigial" in agent.genome
        assert agent.genome["old_skill_vestigial"] == 0.6

    def test_migrate_multiple_agents_same_schema(self) -> None:
        """Test migrating multiple agents to the same schema."""
        from loopengine.discovery.discoverer import migrate_genome
        from loopengine.model.agent import Agent

        agents = [
            Agent(
                id="agent-1",
                name="Worker 1",
                role="worker",
                genome={"speed": 0.9, "strength": 0.3, "old_trait": 0.5},
            ),
            Agent(
                id="agent-2",
                name="Worker 2",
                role="worker",
                genome={"speed": 0.4, "strength": 0.8, "old_trait": 0.2},
            ),
        ]

        new_schema = GenomeSchema(
            role="worker",
            traits={
                "speed": GenomeTrait(name="speed", description="Speed"),
                "strength": GenomeTrait(name="strength", description="Strength"),
                "new_trait": GenomeTrait(name="new_trait", description="New"),
            },
            version=2,
        )

        # Migrate all agents
        for agent in agents:
            result = migrate_genome(agent.genome, new_schema)
            agent.genome = result.migrated_genome

        # Both agents should have same trait keys
        assert set(agents[0].genome.keys()) == set(agents[1].genome.keys())

        # Original values should be preserved
        assert agents[0].genome["speed"] == 0.9
        assert agents[1].genome["speed"] == 0.4

        # Vestigial traits preserved
        assert agents[0].genome["old_trait_vestigial"] == 0.5
        assert agents[1].genome["old_trait_vestigial"] == 0.2

        # New traits added (values will differ due to randomness)
        assert "new_trait" in agents[0].genome
        assert "new_trait" in agents[1].genome


class TestMigrationModuleExports:
    """Tests for migration module exports."""

    def test_import_migrate_genome_from_package(self) -> None:
        """Test migrate_genome can be imported from discovery package."""
        from loopengine.discovery import migrate_genome

        assert callable(migrate_genome)

    def test_import_migration_result_from_package(self) -> None:
        """Test MigrationResult can be imported from discovery package."""
        from loopengine.discovery import MigrationResult

        result = MigrationResult()
        assert result.migrated_genome == {}

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
            trait = schemas["owner"].traits["leadership"]
            assert trait.min_val == 0.0
            assert trait.max_val == 1.0
            assert trait.category == ""


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

"""Tests for the domain description parser."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.domain_parser import (
    AgentTypeSchema,
    DomainParser,
    DomainParserError,
    DomainSchema,
    InteractionSchema,
    ResourceSchema,
)
from loopengine.behaviors.llm_client import BehaviorResponse, LLMClient, LLMQuery


@pytest.fixture
def mock_config() -> LLMConfig:
    """Create a mock LLM config."""
    return LLMConfig(
        llm_provider=LLMProvider.CLAUDE,
        anthropic_api_key=SecretStr("test-api-key"),
        llm_max_tokens=500,
        llm_temperature=0.7,
        llm_timeout=30.0,
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client."""
    return MagicMock(spec=LLMClient)


@pytest.fixture
def sample_schema_response() -> dict:
    """Create a sample valid schema response."""
    return {
        "domain_type": "sandwich shop",
        "description": "A fast-food sandwich shop with made-to-order sandwiches",
        "agent_types": [
            {
                "name": "sandwich_maker",
                "role": "Prepares sandwiches for customers",
                "capabilities": ["make_sandwich", "assemble_ingredients", "wrap_sandwich"],
            },
            {
                "name": "customer",
                "role": "Orders and receives sandwiches",
                "capabilities": ["place_order", "pay", "wait", "receive_order"],
            },
        ],
        "resources": [
            {
                "name": "bread",
                "description": "Base ingredient for sandwiches",
                "consumable": True,
            },
            {
                "name": "sandwich",
                "description": "Completed sandwich product",
                "consumable": True,
            },
        ],
        "interactions": [
            {
                "name": "order_transaction",
                "participants": ["customer", "sandwich_maker"],
                "description": "Customer places an order with sandwich maker",
            },
        ],
    }


class TestDomainSchemaModels:
    """Tests for the Pydantic schema models."""

    def test_agent_type_schema_basic(self) -> None:
        """Test AgentTypeSchema with basic data."""
        agent = AgentTypeSchema(
            name="florist",
            role="Arranges flowers for customers",
            capabilities=["arrange_flowers", "suggest_combinations"],
        )
        assert agent.name == "florist"
        assert agent.role == "Arranges flowers for customers"
        assert len(agent.capabilities) == 2

    def test_agent_type_schema_minimal(self) -> None:
        """Test AgentTypeSchema with minimal data."""
        agent = AgentTypeSchema(name="worker")
        assert agent.name == "worker"
        assert agent.role == ""
        assert agent.capabilities == []

    def test_resource_schema_basic(self) -> None:
        """Test ResourceSchema with basic data."""
        resource = ResourceSchema(
            name="flower",
            description="A single flower for arrangements",
            consumable=True,
        )
        assert resource.name == "flower"
        assert resource.description == "A single flower for arrangements"
        assert resource.consumable is True

    def test_resource_schema_minimal(self) -> None:
        """Test ResourceSchema with minimal data."""
        resource = ResourceSchema(name="item")
        assert resource.name == "item"
        assert resource.consumable is True  # default

    def test_interaction_schema_basic(self) -> None:
        """Test InteractionSchema with basic data."""
        interaction = InteractionSchema(
            name="sale",
            participants=["seller", "buyer"],
            description="Exchange of goods for payment",
        )
        assert interaction.name == "sale"
        assert interaction.participants == ["seller", "buyer"]

    def test_domain_schema_full(self, sample_schema_response: dict) -> None:
        """Test DomainSchema with complete data."""
        schema = DomainSchema(**sample_schema_response)
        assert schema.domain_type == "sandwich shop"
        assert len(schema.agent_types) == 2
        assert len(schema.resources) == 2
        assert len(schema.interactions) == 1

    def test_domain_schema_minimal(self) -> None:
        """Test DomainSchema with minimal data."""
        schema = DomainSchema(domain_type="test domain")
        assert schema.domain_type == "test domain"
        assert schema.agent_types == []
        assert schema.resources == []
        assert schema.interactions == []

    def test_domain_schema_rejects_empty_domain_type(self) -> None:
        """Test DomainSchema validates domain_type is not empty."""
        with pytest.raises(ValueError) as exc_info:
            DomainSchema(domain_type="")
        assert "cannot be empty" in str(exc_info.value)

    def test_domain_schema_strips_whitespace_domain_type(self) -> None:
        """Test DomainSchema strips whitespace from domain_type."""
        schema = DomainSchema(domain_type="  coffee shop  ")
        assert schema.domain_type == "coffee shop"

    def test_domain_schema_rejects_whitespace_only_domain_type(self) -> None:
        """Test DomainSchema rejects whitespace-only domain_type."""
        with pytest.raises(ValueError) as exc_info:
            DomainSchema(domain_type="   ")
        assert "cannot be empty" in str(exc_info.value)


class TestDomainParserInit:
    """Tests for DomainParser initialization."""

    def test_init_with_defaults(self, mock_config: LLMConfig) -> None:
        """Test parser initializes with default components."""
        with patch("loopengine.behaviors.domain_parser.ClaudeClient"):
            with patch(
                "loopengine.behaviors.domain_parser.get_llm_config",
                return_value=mock_config,
            ):
                parser = DomainParser(config=mock_config)
                assert parser._config == mock_config

    def test_init_with_injected_client(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test parser initializes with injected LLM client."""
        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        assert parser._llm_client == mock_llm_client


class TestDomainParserParse:
    """Tests for the parse method."""

    def test_parse_sandwich_shop_description(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
        sample_schema_response: dict,
    ) -> None:
        """Test parsing a sandwich shop description."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(sample_schema_response),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        schema = parser.parse("A sandwich shop where employees make sandwiches for customers")

        assert schema.domain_type == "sandwich shop"
        assert len(schema.agent_types) == 2
        agent_names = [a.name for a in schema.agent_types]
        assert "sandwich_maker" in agent_names
        assert "customer" in agent_names

    def test_parse_simple_description(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test parsing a minimal single-sentence description."""
        minimal_response = {
            "domain_type": "bakery",
            "description": "A small bakery",
            "agent_types": [
                {
                    "name": "baker",
                    "role": "Makes baked goods",
                    "capabilities": ["bake"],
                }
            ],
            "resources": [],
            "interactions": [],
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(minimal_response),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        schema = parser.parse("A bakery")

        assert schema.domain_type == "bakery"
        assert len(schema.agent_types) == 1
        assert schema.agent_types[0].name == "baker"

    def test_parse_detailed_flower_shop_description(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test parsing a detailed flower shop description with multiple agent types."""
        flower_shop_response = {
            "domain_type": "flower shop",
            "description": "A flower shop with delivery service",
            "agent_types": [
                {
                    "name": "florist",
                    "role": "Creates flower arrangements",
                    "capabilities": [
                        "arrange_flowers",
                        "suggest_combinations",
                        "take_order",
                    ],
                },
                {
                    "name": "customer",
                    "role": "Orders flowers",
                    "capabilities": ["browse", "place_order", "pay"],
                },
                {
                    "name": "driver",
                    "role": "Delivers flower orders",
                    "capabilities": ["pick_up_order", "deliver", "confirm_delivery"],
                },
            ],
            "resources": [
                {"name": "flower", "description": "Individual flowers", "consumable": True},
                {"name": "bouquet", "description": "Arranged bouquet", "consumable": True},
            ],
            "interactions": [
                {
                    "name": "order_placement",
                    "participants": ["customer", "florist"],
                    "description": "Customer orders flowers from florist",
                },
                {
                    "name": "delivery",
                    "participants": ["driver", "customer"],
                    "description": "Driver delivers order to customer",
                },
            ],
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(flower_shop_response),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        schema = parser.parse(
            "A flower shop where florists create arrangements for customers. "
            "Drivers deliver orders to customers' homes."
        )

        assert schema.domain_type == "flower shop"
        assert len(schema.agent_types) == 3
        agent_names = [a.name for a in schema.agent_types]
        assert "florist" in agent_names
        assert "customer" in agent_names
        assert "driver" in agent_names

    def test_parse_empty_description_raises_error(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that empty description raises DomainParserError."""
        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(DomainParserError) as exc_info:
            parser.parse("")

        assert "cannot be empty" in str(exc_info.value)

    def test_parse_whitespace_description_raises_error(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that whitespace-only description raises DomainParserError."""
        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(DomainParserError) as exc_info:
            parser.parse("   \n\t  ")

        assert "cannot be empty" in str(exc_info.value)

    def test_parse_invalid_json_raises_error(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that invalid JSON response raises DomainParserError."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning="This is not valid JSON at all",
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(DomainParserError) as exc_info:
            parser.parse("A test domain")

        assert "Invalid JSON" in str(exc_info.value)

    def test_parse_handles_llm_error(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that LLM errors are wrapped in DomainParserError."""
        mock_llm_client.query.side_effect = Exception("LLM connection failed")

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(DomainParserError) as exc_info:
            parser.parse("A test domain")

        assert "Failed to parse" in str(exc_info.value)

    def test_parse_verifies_llm_prompt_structure(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that the correct prompt structure is sent to LLM."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(
                {"domain_type": "test", "agent_types": [], "resources": [], "interactions": []}
            ),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        parser.parse("A coffee shop")

        # Verify the query was called
        mock_llm_client.query.assert_called_once()
        call_args = mock_llm_client.query.call_args
        query: LLMQuery = call_args[0][0]

        # Verify system message asks for JSON output
        assert "JSON" in query.system_message
        assert "domain_type" in query.system_message
        assert "agent_types" in query.system_message

        # Verify prompt includes the description
        assert "coffee shop" in query.prompt


class TestDomainParserJsonExtraction:
    """Tests for JSON extraction from various response formats."""

    def test_extracts_json_from_code_block(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test extraction from markdown code block."""
        response_with_code_block = """Here's the schema:

```json
{
    "domain_type": "cafe",
    "description": "A coffee cafe",
    "agent_types": [{"name": "barista", "role": "Makes coffee", "capabilities": ["brew"]}],
    "resources": [],
    "interactions": []
}
```

This represents a simple cafe."""

        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=response_with_code_block,
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        schema = parser.parse("A coffee cafe")

        assert schema.domain_type == "cafe"
        assert len(schema.agent_types) == 1

    def test_extracts_json_from_generic_code_block(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test extraction from generic code block without json tag."""
        response_with_code_block = """```
{
    "domain_type": "gym",
    "description": "A fitness gym",
    "agent_types": [],
    "resources": [],
    "interactions": []
}
```"""

        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=response_with_code_block,
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        schema = parser.parse("A gym")

        assert schema.domain_type == "gym"

    def test_extracts_embedded_json(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test extraction of JSON embedded in surrounding text."""
        embedded_json = {
            "domain_type": "library",
            "description": "A public library",
            "agent_types": [
                {"name": "librarian", "role": "Manages books", "capabilities": ["check_out_book"]}
            ],
            "resources": [],
            "interactions": [],
        }
        response_with_embedded = (
            "Based on your description, I've extracted the following schema:\n\n"
            + json.dumps(embedded_json)
            + "\n\nHope this helps!"
        )

        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=response_with_embedded,
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        schema = parser.parse("A library")

        assert schema.domain_type == "library"
        assert len(schema.agent_types) == 1


class TestDomainParserParseWithMetadata:
    """Tests for the parse_with_metadata method."""

    def test_parse_with_metadata_returns_tuple(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
        sample_schema_response: dict,
    ) -> None:
        """Test parse_with_metadata returns schema and metadata tuple."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(sample_schema_response),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        schema, metadata = parser.parse_with_metadata("A sandwich shop")

        assert isinstance(schema, DomainSchema)
        assert isinstance(metadata, dict)
        assert "latency_ms" in metadata
        assert metadata["latency_ms"] >= 0

    def test_parse_with_metadata_includes_counts(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
        sample_schema_response: dict,
    ) -> None:
        """Test parse_with_metadata includes extraction counts."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(sample_schema_response),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        _, metadata = parser.parse_with_metadata("A sandwich shop")

        assert metadata["agent_types_extracted"] == 2
        assert metadata["resources_extracted"] == 2
        assert metadata["interactions_extracted"] == 1


class TestDomainParserImport:
    """Tests for import and instantiation."""

    def test_import_from_behaviors_package(self) -> None:
        """Test all classes can be imported from behaviors package."""
        from loopengine.behaviors import (
            AgentTypeSchema,
            DomainParser,
            DomainParserError,
            DomainSchema,
            InteractionSchema,
            ResourceSchema,
        )

        assert DomainParser is not None
        assert DomainParserError is not None
        assert DomainSchema is not None
        assert AgentTypeSchema is not None
        assert ResourceSchema is not None
        assert InteractionSchema is not None

    def test_full_integration_with_mocked_llm(self, mock_config: LLMConfig) -> None:
        """Test full integration with mocked LLM client."""
        from loopengine.behaviors import DomainParser, DomainSchema

        mock_client = MagicMock(spec=LLMClient)
        mock_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(
                {
                    "domain_type": "restaurant",
                    "description": "A fine dining restaurant",
                    "agent_types": [
                        {
                            "name": "chef",
                            "role": "Prepares meals",
                            "capabilities": ["cook", "plate"],
                        },
                        {
                            "name": "waiter",
                            "role": "Serves customers",
                            "capabilities": ["take_order", "serve"],
                        },
                        {
                            "name": "customer",
                            "role": "Dines at restaurant",
                            "capabilities": ["order", "eat", "pay"],
                        },
                    ],
                    "resources": [
                        {"name": "meal", "description": "Prepared food", "consumable": True}
                    ],
                    "interactions": [],
                }
            ),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_client, config=mock_config)
        schema = parser.parse(
            "A fine dining restaurant where chefs prepare gourmet meals and waiters serve customers"
        )

        assert isinstance(schema, DomainSchema)
        assert schema.domain_type == "restaurant"
        assert len(schema.agent_types) == 3

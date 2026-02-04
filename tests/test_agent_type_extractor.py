"""Tests for the agent type extractor."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from loopengine.behaviors.agent_type_extractor import (
    AgentType,
    AgentTypeExtractor,
    AgentTypeExtractorError,
)
from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.llm_client import BehaviorResponse, LLMClient


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


class TestAgentTypeModel:
    """Tests for the AgentType Pydantic model."""

    def test_agent_type_basic(self) -> None:
        """Test AgentType with basic data."""
        agent = AgentType(
            name="florist",
            role="Arranges flowers for customers",
            capabilities=["make_bouquet", "arrange_flowers"],
            default_state={"status": "idle", "energy": 100},
        )
        assert agent.name == "florist"
        assert agent.role == "Arranges flowers for customers"
        assert len(agent.capabilities) == 2
        assert agent.default_state["status"] == "idle"

    def test_agent_type_minimal(self) -> None:
        """Test AgentType with minimal data."""
        agent = AgentType(name="worker")
        assert agent.name == "worker"
        assert agent.role == ""
        assert agent.capabilities == []
        assert agent.default_state == {}

    def test_agent_type_name_validation_empty(self) -> None:
        """Test AgentType rejects empty name."""
        with pytest.raises(ValueError) as exc_info:
            AgentType(name="")
        assert "cannot be empty" in str(exc_info.value)

    def test_agent_type_name_normalization(self) -> None:
        """Test AgentType normalizes name to snake_case."""
        agent = AgentType(name="Sandwich Maker")
        assert agent.name == "sandwich_maker"

        agent2 = AgentType(name="delivery-driver")
        assert agent2.name == "delivery_driver"

    def test_agent_type_capabilities_normalization(self) -> None:
        """Test AgentType normalizes capabilities to snake_case."""
        agent = AgentType(
            name="worker",
            capabilities=["Make Bouquet", "take-order", "  deliver package  "],
        )
        assert agent.capabilities == ["make_bouquet", "take_order", "deliver_package"]

    def test_agent_type_filters_empty_capabilities(self) -> None:
        """Test AgentType filters out empty capabilities."""
        agent = AgentType(
            name="worker",
            capabilities=["make_bouquet", "", "  ", "deliver"],
        )
        assert agent.capabilities == ["make_bouquet", "deliver"]

    def test_agent_type_with_complex_default_state(self) -> None:
        """Test AgentType with complex default state."""
        agent = AgentType(
            name="customer",
            default_state={
                "status": "browsing",
                "satisfaction": 50,
                "inventory": [],
                "orders": [],
            },
        )
        assert agent.default_state["satisfaction"] == 50
        assert agent.default_state["inventory"] == []


class TestAgentTypeExtractorInit:
    """Tests for AgentTypeExtractor initialization."""

    def test_init_with_injected_client(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test extractor initializes with injected LLM client."""
        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        assert extractor._llm_client == mock_llm_client


class TestAgentTypeExtractorExtract:
    """Tests for the extract method."""

    def test_extract_florists_make_bouquets(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test: 'florists make bouquets' -> florist with make_bouquet capability."""
        response = {
            "agent_types": [
                {
                    "name": "florist",
                    "role": "Creates flower arrangements and bouquets",
                    "capabilities": ["make_bouquet", "arrange_flowers", "take_order"],
                    "default_state": {"status": "idle", "energy": 100},
                    "extraction_type": "explicit",
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("florists make bouquets")

        assert len(agents) == 1
        assert agents[0].name == "florist"
        assert "make_bouquet" in agents[0].capabilities
        assert agents[0].default_state["status"] == "idle"

    def test_extract_customers_order_and_pay(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test: 'customers order and pay' -> customer with order, pay capabilities."""
        response = {
            "agent_types": [
                {
                    "name": "customer",
                    "role": "Orders products and makes payments",
                    "capabilities": ["order", "pay", "browse"],
                    "default_state": {"status": "browsing", "satisfaction": 50},
                    "extraction_type": "explicit",
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("customers order and pay")

        assert len(agents) == 1
        assert agents[0].name == "customer"
        assert "order" in agents[0].capabilities
        assert "pay" in agents[0].capabilities

    def test_extract_implied_agents(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test: 'deliveries happen' -> delivery_driver extracted as implied."""
        response = {
            "agent_types": [
                {
                    "name": "delivery_driver",
                    "role": "Delivers packages to customers",
                    "capabilities": ["pick_up_package", "deliver", "confirm_delivery"],
                    "default_state": {"status": "idle", "location": "base"},
                    "extraction_type": "implied",
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("deliveries happen")

        assert len(agents) == 1
        assert agents[0].name == "delivery_driver"
        assert "deliver" in agents[0].capabilities

    def test_extract_multiple_agent_types(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test extracting multiple agent types from a description."""
        response = {
            "agent_types": [
                {
                    "name": "florist",
                    "role": "Creates flower arrangements",
                    "capabilities": ["make_bouquet", "arrange_flowers"],
                    "default_state": {"status": "idle"},
                },
                {
                    "name": "customer",
                    "role": "Orders flowers",
                    "capabilities": ["place_order", "pay", "receive_flowers"],
                    "default_state": {"status": "browsing"},
                },
                {
                    "name": "delivery_driver",
                    "role": "Delivers flower orders",
                    "capabilities": ["pick_up", "deliver"],
                    "default_state": {"status": "idle", "location": "base"},
                },
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        description = (
            "A flower shop where florists make bouquets. "
            "Customers order flowers and drivers deliver them."
        )
        agents = extractor.extract(description)

        assert len(agents) == 3
        names = [a.name for a in agents]
        assert "florist" in names
        assert "customer" in names
        assert "delivery_driver" in names

    def test_extract_all_agent_types_have_required_fields(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Verify all extracted agent types have required fields."""
        response = {
            "agent_types": [
                {
                    "name": "barista",
                    "role": "Makes coffee drinks",
                    "capabilities": ["make_coffee", "steam_milk"],
                    "default_state": {"status": "ready"},
                },
                {
                    "name": "cashier",
                    "role": "Takes orders and payments",
                    "capabilities": ["take_order", "process_payment"],
                    "default_state": {"status": "available"},
                },
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("A coffee shop with baristas and cashiers")

        for agent in agents:
            assert agent.name, "Agent must have a name"
            assert hasattr(agent, "role"), "Agent must have role attribute"
            assert hasattr(agent, "capabilities"), "Agent must have capabilities attribute"
            assert hasattr(agent, "default_state"), "Agent must have default_state attribute"
            assert isinstance(agent.capabilities, list), "capabilities must be a list"
            assert isinstance(agent.default_state, dict), "default_state must be a dict"

    def test_extract_empty_description_raises_error(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that empty description raises AgentTypeExtractorError."""
        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(AgentTypeExtractorError) as exc_info:
            extractor.extract("")

        assert "cannot be empty" in str(exc_info.value)

    def test_extract_whitespace_description_raises_error(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that whitespace-only description raises AgentTypeExtractorError."""
        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(AgentTypeExtractorError) as exc_info:
            extractor.extract("   \n\t  ")

        assert "cannot be empty" in str(exc_info.value)

    def test_extract_invalid_json_raises_error(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that invalid JSON response raises AgentTypeExtractorError."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning="This is not valid JSON at all",
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(AgentTypeExtractorError) as exc_info:
            extractor.extract("A test domain")

        assert "Invalid JSON" in str(exc_info.value)

    def test_extract_handles_llm_error(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that LLM errors are wrapped in AgentTypeExtractorError."""
        mock_llm_client.query.side_effect = Exception("LLM connection failed")

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(AgentTypeExtractorError) as exc_info:
            extractor.extract("A test domain")

        assert "Failed to extract" in str(exc_info.value)


class TestAgentTypeExtractorDefaultStates:
    """Tests for default state generation."""

    def test_generates_customer_defaults(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that customer-like agents get customer defaults."""
        response = {
            "agent_types": [
                {
                    "name": "customer",
                    "role": "Shops at the store",
                    "capabilities": ["browse", "buy"],
                    # No default_state provided
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("customers shop at the store")

        assert agents[0].default_state["status"] == "browsing"
        assert "satisfaction" in agents[0].default_state

    def test_generates_driver_defaults(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that driver-like agents get driver defaults."""
        response = {
            "agent_types": [
                {
                    "name": "delivery_driver",
                    "role": "Delivers packages",
                    "capabilities": ["deliver"],
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("drivers deliver packages")

        assert agents[0].default_state["status"] == "idle"
        assert "location" in agents[0].default_state

    def test_generates_worker_defaults(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that worker-like agents get worker defaults."""
        response = {
            "agent_types": [
                {
                    "name": "sandwich_maker",
                    "role": "Makes sandwiches",
                    "capabilities": ["make_sandwich"],
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("sandwich makers make sandwiches")

        assert agents[0].default_state["status"] == "idle"
        assert "energy" in agents[0].default_state
        assert "tasks_completed" in agents[0].default_state


class TestAgentTypeExtractorJsonExtraction:
    """Tests for JSON extraction from various response formats."""

    def test_extracts_json_from_code_block(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test extraction from markdown code block."""
        response_with_code_block = """Here are the extracted agent types:

```json
{
    "agent_types": [
        {
            "name": "barista",
            "role": "Makes coffee",
            "capabilities": ["brew_coffee"],
            "default_state": {"status": "ready"}
        }
    ]
}
```

Hope this helps!"""

        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=response_with_code_block,
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("A coffee cafe")

        assert len(agents) == 1
        assert agents[0].name == "barista"

    def test_extracts_embedded_json(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test extraction of JSON embedded in surrounding text."""
        embedded = {
            "agent_types": [
                {
                    "name": "librarian",
                    "role": "Manages books",
                    "capabilities": ["check_out_book"],
                    "default_state": {"status": "available"},
                }
            ]
        }
        response_with_embedded = (
            "Based on your description:\n\n" + json.dumps(embedded) + "\n\nDone!"
        )

        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=response_with_embedded,
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("A library")

        assert len(agents) == 1
        assert agents[0].name == "librarian"


class TestAgentTypeExtractorWithMetadata:
    """Tests for the extract_with_metadata method."""

    def test_extract_with_metadata_returns_tuple(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test extract_with_metadata returns agents and metadata tuple."""
        response = {
            "agent_types": [
                {
                    "name": "worker",
                    "role": "Does work",
                    "capabilities": ["work"],
                    "default_state": {"status": "idle"},
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents, metadata = extractor.extract_with_metadata("workers do work")

        assert isinstance(agents, list)
        assert len(agents) == 1
        assert isinstance(metadata, dict)
        assert "latency_ms" in metadata
        assert metadata["latency_ms"] >= 0

    def test_extract_with_metadata_includes_counts(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test extract_with_metadata includes extraction counts."""
        response = {
            "agent_types": [
                {
                    "name": "worker1",
                    "role": "Works",
                    "capabilities": ["work", "rest"],
                    "default_state": {},
                },
                {
                    "name": "worker2",
                    "role": "Also works",
                    "capabilities": ["work"],
                    "default_state": {},
                },
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        _, metadata = extractor.extract_with_metadata("two workers")

        assert metadata["agent_types_extracted"] == 2
        assert metadata["total_capabilities"] == 3


class TestAgentTypeExtractorImport:
    """Tests for import and instantiation."""

    def test_import_from_behaviors_package(self) -> None:
        """Test all classes can be imported from behaviors package."""
        from loopengine.behaviors import (
            AgentType,
            AgentTypeExtractor,
            AgentTypeExtractorError,
        )

        assert AgentType is not None
        assert AgentTypeExtractor is not None
        assert AgentTypeExtractorError is not None

    def test_full_integration_with_mocked_llm(self, mock_config: LLMConfig) -> None:
        """Test full integration with mocked LLM client."""
        from loopengine.behaviors import AgentType, AgentTypeExtractor

        mock_client = MagicMock(spec=LLMClient)
        mock_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(
                {
                    "agent_types": [
                        {
                            "name": "chef",
                            "role": "Prepares meals",
                            "capabilities": ["cook", "plate", "garnish"],
                            "default_state": {"status": "idle", "energy": 100},
                        },
                        {
                            "name": "waiter",
                            "role": "Serves customers",
                            "capabilities": ["take_order", "serve", "clear_table"],
                            "default_state": {"status": "available"},
                        },
                    ]
                }
            ),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_client, config=mock_config)
        agents = extractor.extract("A restaurant where chefs cook and waiters serve")

        assert len(agents) == 2
        assert all(isinstance(a, AgentType) for a in agents)
        names = [a.name for a in agents]
        assert "chef" in names
        assert "waiter" in names


class TestAgentTypeExtractorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_skips_agents_with_empty_names(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test that agents with empty names are skipped."""
        response = {
            "agent_types": [
                {
                    "name": "",
                    "role": "Invalid agent",
                    "capabilities": [],
                },
                {
                    "name": "valid_agent",
                    "role": "Valid agent",
                    "capabilities": ["work"],
                },
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("test domain")

        assert len(agents) == 1
        assert agents[0].name == "valid_agent"

    def test_handles_missing_agent_types_key(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test handling response without agent_types key."""
        response = {"domain": "test"}  # Missing agent_types
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)
        agents = extractor.extract("test domain")

        assert agents == []

    def test_handles_non_list_agent_types(
        self, mock_config: LLMConfig, mock_llm_client: MagicMock
    ) -> None:
        """Test handling response with non-list agent_types."""
        response = {"agent_types": "not a list"}
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response),
            metadata={},
        )

        extractor = AgentTypeExtractor(llm_client=mock_llm_client, config=mock_config)

        with pytest.raises(AgentTypeExtractorError) as exc_info:
            extractor.extract("test domain")

        assert "must be a list" in str(exc_info.value)

"""Tests for the interaction extractor."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.interaction_extractor import (
    Interaction,
    InteractionExtractor,
    InteractionExtractorError,
)
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


@pytest.fixture
def sample_interactions_response() -> dict:
    """Create a sample valid interactions response."""
    return {
        "interactions": [
            {
                "name": "customer_order",
                "source_agent": "customer",
                "target": "florist",
                "action_type": "request",
                "conditions": ["during_business_hours"],
                "description": "Customer places an order with florist",
            },
            {
                "name": "florist_fulfill",
                "source_agent": "florist",
                "target": "flowers",
                "action_type": "consume",
                "conditions": ["sufficient_inventory"],
                "description": "Florist uses flowers to create arrangement",
            },
        ]
    }


class TestInteractionModel:
    """Tests for the Interaction Pydantic model."""

    def test_interaction_basic(self) -> None:
        """Test Interaction with basic data."""
        interaction = Interaction(
            name="customer_order",
            source_agent="customer",
            target="florist",
            action_type="request",
            conditions=["during_hours"],
            description="Customer places order",
        )
        assert interaction.name == "customer_order"
        assert interaction.source_agent == "customer"
        assert interaction.target == "florist"
        assert interaction.action_type == "request"
        assert interaction.conditions == ["during_hours"]

    def test_interaction_minimal(self) -> None:
        """Test Interaction with minimal data."""
        interaction = Interaction(
            name="test_interaction",
            source_agent="agent_a",
            target="agent_b",
        )
        assert interaction.name == "test_interaction"
        assert interaction.action_type == "interact"
        assert interaction.conditions == []
        assert interaction.description == ""

    def test_interaction_name_normalized(self) -> None:
        """Test that interaction names are normalized to snake_case."""
        interaction = Interaction(
            name="Customer Order",
            source_agent="customer",
            target="florist",
        )
        assert interaction.name == "customer_order"

        interaction2 = Interaction(
            name="place-order",
            source_agent="customer",
            target="florist",
        )
        assert interaction2.name == "place_order"

    def test_interaction_source_agent_normalized(self) -> None:
        """Test that source_agent is normalized to snake_case."""
        interaction = Interaction(
            name="test",
            source_agent="Flower Customer",
            target="florist",
        )
        assert interaction.source_agent == "flower_customer"

    def test_interaction_target_normalized(self) -> None:
        """Test that target is normalized to snake_case."""
        interaction = Interaction(
            name="test",
            source_agent="customer",
            target="Delivery Driver",
        )
        assert interaction.target == "delivery_driver"

    def test_interaction_empty_name_raises(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Interaction(
                name="",
                source_agent="customer",
                target="florist",
            )

    def test_interaction_empty_source_agent_raises(self) -> None:
        """Test that empty source_agent raises ValueError."""
        with pytest.raises(ValueError, match="source_agent cannot be empty"):
            Interaction(
                name="test",
                source_agent="",
                target="florist",
            )

    def test_interaction_empty_target_raises(self) -> None:
        """Test that empty target raises ValueError."""
        with pytest.raises(ValueError, match="target cannot be empty"):
            Interaction(
                name="test",
                source_agent="customer",
                target="",
            )


class TestInteractionExtractor:
    """Tests for the InteractionExtractor class."""

    def test_extract_basic(
        self, mock_llm_client: MagicMock, sample_interactions_response: dict
    ) -> None:
        """Test basic interaction extraction."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(sample_interactions_response),
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Customers order from florists")

        assert len(interactions) == 2
        assert interactions[0].name == "customer_order"
        assert interactions[0].source_agent == "customer"
        assert interactions[0].target == "florist"
        assert interactions[0].action_type == "request"

    def test_extract_customer_florist_interaction(self, mock_llm_client: MagicMock) -> None:
        """Test: 'customers order from florists' -> customer-florist interaction."""
        response = {
            "interactions": [
                {
                    "name": "customer_order",
                    "source_agent": "customer",
                    "target": "florist",
                    "action_type": "request",
                    "conditions": [],
                    "description": "Customer places order with florist",
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("customers order from florists")

        assert len(interactions) == 1
        assert interactions[0].source_agent == "customer"
        assert interactions[0].target == "florist"
        assert interactions[0].action_type == "request"

    def test_extract_references_valid_agent_types(self, mock_llm_client: MagicMock) -> None:
        """Test that interactions reference valid agent types."""
        response = {
            "interactions": [
                {
                    "name": "order_food",
                    "source_agent": "customer",
                    "target": "waiter",
                    "action_type": "request",
                },
                {
                    "name": "prepare_food",
                    "source_agent": "chef",
                    "target": "ingredients",
                    "action_type": "consume",
                },
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Restaurant with customers, waiters, and chefs")

        assert len(interactions) == 2
        source_agents = {i.source_agent for i in interactions}
        assert source_agents == {"customer", "chef"}

    def test_extract_empty_description_raises(self, mock_llm_client: MagicMock) -> None:
        """Test that empty description raises error."""
        extractor = InteractionExtractor(llm_client=mock_llm_client)

        with pytest.raises(InteractionExtractorError, match="Description cannot be empty"):
            extractor.extract("")

        with pytest.raises(InteractionExtractorError, match="Description cannot be empty"):
            extractor.extract("   ")

    def test_extract_with_json_code_block(self, mock_llm_client: MagicMock) -> None:
        """Test extraction from JSON wrapped in code block."""
        response = {
            "interactions": [
                {
                    "name": "test",
                    "source_agent": "a",
                    "target": "b",
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=f"```json\n{json.dumps(response)}\n```",
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Test domain")

        assert len(interactions) == 1

    def test_extract_with_generic_code_block(self, mock_llm_client: MagicMock) -> None:
        """Test extraction from JSON wrapped in generic code block."""
        response = {
            "interactions": [
                {
                    "name": "test",
                    "source_agent": "a",
                    "target": "b",
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=f"```\n{json.dumps(response)}\n```",
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Test domain")

        assert len(interactions) == 1

    def test_extract_with_embedded_json(self, mock_llm_client: MagicMock) -> None:
        """Test extraction from JSON embedded in text."""
        response = {
            "interactions": [
                {
                    "name": "test",
                    "source_agent": "a",
                    "target": "b",
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=f"Here is the result: {json.dumps(response)} done.",
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Test domain")

        assert len(interactions) == 1

    def test_extract_skips_invalid_interactions(self, mock_llm_client: MagicMock) -> None:
        """Test that interactions with missing required fields are skipped."""
        response = {
            "interactions": [
                {"name": "valid", "source_agent": "a", "target": "b"},
                {"name": "missing_source", "target": "b"},  # Missing source_agent
                {"name": "missing_target", "source_agent": "a"},  # Missing target
                {"source_agent": "a", "target": "b"},  # Missing name
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Test domain")

        assert len(interactions) == 1
        assert interactions[0].name == "valid"

    def test_extract_handles_non_list_conditions(self, mock_llm_client: MagicMock) -> None:
        """Test that non-list conditions are converted to list."""
        response = {
            "interactions": [
                {
                    "name": "test",
                    "source_agent": "a",
                    "target": "b",
                    "conditions": "single_condition",
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Test domain")

        assert interactions[0].conditions == ["single_condition"]

    def test_extract_invalid_json_raises(self, mock_llm_client: MagicMock) -> None:
        """Test that invalid JSON raises error."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning="not valid json at all",
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)

        with pytest.raises(InteractionExtractorError, match="Invalid JSON"):
            extractor.extract("Test domain")

    def test_extract_non_dict_response_raises(self, mock_llm_client: MagicMock) -> None:
        """Test that non-dict JSON response raises error."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning="[]",
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)

        with pytest.raises(InteractionExtractorError, match="Expected JSON object"):
            extractor.extract("Test domain")

    def test_extract_non_list_interactions_raises(self, mock_llm_client: MagicMock) -> None:
        """Test that non-list interactions field raises error."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning='{"interactions": "not a list"}',
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)

        with pytest.raises(InteractionExtractorError, match="interactions must be a list"):
            extractor.extract("Test domain")

    def test_extract_with_metadata(
        self, mock_llm_client: MagicMock, sample_interactions_response: dict
    ) -> None:
        """Test extraction with metadata."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(sample_interactions_response),
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions, metadata = extractor.extract_with_metadata("Flower shop")

        assert len(interactions) == 2
        assert "latency_ms" in metadata
        assert metadata["interactions_extracted"] == 2
        assert "action_type_breakdown" in metadata
        assert metadata["action_type_breakdown"]["request"] == 1
        assert metadata["action_type_breakdown"]["consume"] == 1
        assert metadata["unique_source_agents"] == 2
        assert metadata["unique_targets"] == 2

    def test_extract_all_action_types(self, mock_llm_client: MagicMock) -> None:
        """Test extraction of various action types."""
        response = {
            "interactions": [
                {"name": "i1", "source_agent": "a", "target": "b", "action_type": "request"},
                {"name": "i2", "source_agent": "a", "target": "b", "action_type": "provide"},
                {"name": "i3", "source_agent": "a", "target": "b", "action_type": "consume"},
                {"name": "i4", "source_agent": "a", "target": "b", "action_type": "transfer"},
                {"name": "i5", "source_agent": "a", "target": "b", "action_type": "query"},
                {"name": "i6", "source_agent": "a", "target": "b", "action_type": "update"},
                {"name": "i7", "source_agent": "a", "target": "b", "action_type": "create"},
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Complex domain")

        action_types = {i.action_type for i in interactions}
        expected = {"request", "provide", "consume", "transfer", "query", "update", "create"}
        assert action_types == expected

    def test_extract_uses_action_field_as_fallback(self, mock_llm_client: MagicMock) -> None:
        """Test that action field is used when reasoning is empty."""
        response = {"interactions": [{"name": "test", "source_agent": "a", "target": "b"}]}
        mock_llm_client.query.return_value = BehaviorResponse(
            action=json.dumps(response),
            reasoning="",
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Test domain")

        assert len(interactions) == 1
        assert interactions[0].name == "test"

    def test_llm_exception_wrapped(self, mock_llm_client: MagicMock) -> None:
        """Test that LLM exceptions are wrapped in InteractionExtractorError."""
        mock_llm_client.query.side_effect = RuntimeError("API error")

        extractor = InteractionExtractor(llm_client=mock_llm_client)

        with pytest.raises(InteractionExtractorError, match="Failed to extract interactions"):
            extractor.extract("Test domain")

    def test_extract_with_conditions(self, mock_llm_client: MagicMock) -> None:
        """Test that conditions are correctly extracted."""
        response = {
            "interactions": [
                {
                    "name": "order",
                    "source_agent": "customer",
                    "target": "shop",
                    "action_type": "request",
                    "conditions": [
                        "target_available",
                        "within_operating_hours",
                        "sufficient_inventory",
                    ],
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = InteractionExtractor(llm_client=mock_llm_client)
        interactions = extractor.extract("Shop with operating hours")

        assert len(interactions[0].conditions) == 3
        assert "target_available" in interactions[0].conditions
        assert "within_operating_hours" in interactions[0].conditions

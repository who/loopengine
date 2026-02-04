"""Tests for the resource extractor."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.llm_client import BehaviorResponse, LLMClient
from loopengine.behaviors.resource_extractor import (
    Resource,
    ResourceExtractor,
    ResourceExtractorError,
)


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
def sample_resources_response() -> dict:
    """Create a sample valid resources response."""
    return {
        "resources": [
            {
                "name": "flowers",
                "resource_type": "item",
                "description": "Fresh cut flowers in inventory",
                "initial_quantity": 100,
                "constraints": ["perishable"],
            },
            {
                "name": "roses",
                "resource_type": "item",
                "description": "Premium roses with limited stock",
                "initial_quantity": 20,
                "constraints": ["limited_stock", "premium"],
            },
            {
                "name": "counter",
                "resource_type": "location",
                "description": "Customer service counter",
                "initial_quantity": 1,
                "constraints": [],
            },
        ]
    }


class TestResourceModel:
    """Tests for the Resource Pydantic model."""

    def test_resource_basic(self) -> None:
        """Test Resource with basic data."""
        resource = Resource(
            name="flowers",
            resource_type="item",
            description="Cut flowers",
            initial_quantity=100,
            constraints=["perishable"],
        )
        assert resource.name == "flowers"
        assert resource.resource_type == "item"
        assert resource.initial_quantity == 100
        assert resource.constraints == ["perishable"]

    def test_resource_minimal(self) -> None:
        """Test Resource with minimal data."""
        resource = Resource(name="bread")
        assert resource.name == "bread"
        assert resource.resource_type == "item"
        assert resource.description == ""
        assert resource.initial_quantity == -1
        assert resource.constraints == []

    def test_resource_name_normalized(self) -> None:
        """Test that resource names are normalized to snake_case."""
        resource = Resource(name="Fresh Flowers")
        assert resource.name == "fresh_flowers"

        resource2 = Resource(name="bread-loaf")
        assert resource2.name == "bread_loaf"

    def test_resource_empty_name_raises(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            Resource(name="")

        with pytest.raises(ValueError, match="name cannot be empty"):
            Resource(name="   ")

    def test_resource_type_normalized(self) -> None:
        """Test that invalid resource types default to item."""
        resource = Resource(name="test", resource_type="unknown")
        assert resource.resource_type == "item"

        resource2 = Resource(name="test", resource_type="ITEM")
        assert resource2.resource_type == "item"

    def test_resource_type_valid_types(self) -> None:
        """Test all valid resource types."""
        for rtype in ["item", "location", "queue", "service"]:
            resource = Resource(name="test", resource_type=rtype)
            assert resource.resource_type == rtype


class TestResourceExtractor:
    """Tests for the ResourceExtractor class."""

    def test_extract_basic(
        self, mock_llm_client: MagicMock, sample_resources_response: dict
    ) -> None:
        """Test basic resource extraction."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(sample_resources_response),
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("A flower shop with roses and flowers")

        assert len(resources) == 3
        assert resources[0].name == "flowers"
        assert resources[0].resource_type == "item"
        assert resources[0].initial_quantity == 100
        assert resources[1].name == "roses"
        assert resources[1].constraints == ["limited_stock", "premium"]
        assert resources[2].name == "counter"
        assert resources[2].resource_type == "location"

    def test_extract_flowers_from_inventory(self, mock_llm_client: MagicMock) -> None:
        """Test: 'flowers in inventory' -> flowers resource extracted."""
        response = {
            "resources": [
                {
                    "name": "flowers",
                    "resource_type": "item",
                    "description": "Flowers in inventory",
                    "initial_quantity": -1,
                    "constraints": [],
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("flowers in inventory")

        assert len(resources) == 1
        assert resources[0].name == "flowers"
        assert resources[0].resource_type == "item"

    def test_extract_limited_stock_constraint(self, mock_llm_client: MagicMock) -> None:
        """Test: 'limited stock of roses' -> rose resource with quantity constraint."""
        response = {
            "resources": [
                {
                    "name": "roses",
                    "resource_type": "item",
                    "description": "Limited stock roses",
                    "initial_quantity": 50,
                    "constraints": ["limited_stock"],
                }
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("limited stock of roses")

        assert len(resources) == 1
        assert resources[0].name == "roses"
        assert "limited_stock" in resources[0].constraints

    def test_extract_empty_description_raises(self, mock_llm_client: MagicMock) -> None:
        """Test that empty description raises error."""
        extractor = ResourceExtractor(llm_client=mock_llm_client)

        with pytest.raises(ResourceExtractorError, match="Description cannot be empty"):
            extractor.extract("")

        with pytest.raises(ResourceExtractorError, match="Description cannot be empty"):
            extractor.extract("   ")

    def test_extract_with_json_code_block(self, mock_llm_client: MagicMock) -> None:
        """Test extraction from JSON wrapped in code block."""
        response = {"resources": [{"name": "bread", "resource_type": "item"}]}
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=f"```json\n{json.dumps(response)}\n```",
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("A bakery with bread")

        assert len(resources) == 1
        assert resources[0].name == "bread"

    def test_extract_with_generic_code_block(self, mock_llm_client: MagicMock) -> None:
        """Test extraction from JSON wrapped in generic code block."""
        response = {"resources": [{"name": "coffee", "resource_type": "item"}]}
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=f"```\n{json.dumps(response)}\n```",
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("A coffee shop")

        assert len(resources) == 1
        assert resources[0].name == "coffee"

    def test_extract_with_embedded_json(self, mock_llm_client: MagicMock) -> None:
        """Test extraction from JSON embedded in text."""
        response = {"resources": [{"name": "tea", "resource_type": "item"}]}
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=f"Here is the result: {json.dumps(response)} done.",
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("A tea shop")

        assert len(resources) == 1
        assert resources[0].name == "tea"

    def test_extract_skips_invalid_resources(self, mock_llm_client: MagicMock) -> None:
        """Test that resources with missing names are skipped."""
        response = {
            "resources": [
                {"name": "valid", "resource_type": "item"},
                {"resource_type": "item"},  # Missing name
                {"name": "", "resource_type": "item"},  # Empty name
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("Test domain")

        assert len(resources) == 1
        assert resources[0].name == "valid"

    def test_extract_handles_non_list_constraints(self, mock_llm_client: MagicMock) -> None:
        """Test that non-list constraints are converted to list."""
        response = {
            "resources": [{"name": "item", "resource_type": "item", "constraints": "single"}]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("Test domain")

        assert resources[0].constraints == ["single"]

    def test_extract_invalid_json_raises(self, mock_llm_client: MagicMock) -> None:
        """Test that invalid JSON raises error."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning="not valid json at all",
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)

        with pytest.raises(ResourceExtractorError, match="Invalid JSON"):
            extractor.extract("Test domain")

    def test_extract_non_dict_response_raises(self, mock_llm_client: MagicMock) -> None:
        """Test that non-dict JSON response raises error."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning="[]",  # Array instead of object
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)

        with pytest.raises(ResourceExtractorError, match="Expected JSON object"):
            extractor.extract("Test domain")

    def test_extract_non_list_resources_raises(self, mock_llm_client: MagicMock) -> None:
        """Test that non-list resources field raises error."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning='{"resources": "not a list"}',
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)

        with pytest.raises(ResourceExtractorError, match="resources must be a list"):
            extractor.extract("Test domain")

    def test_extract_with_metadata(
        self, mock_llm_client: MagicMock, sample_resources_response: dict
    ) -> None:
        """Test extraction with metadata."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(sample_resources_response),
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources, metadata = extractor.extract_with_metadata("A flower shop")

        assert len(resources) == 3
        assert "latency_ms" in metadata
        assert metadata["resources_extracted"] == 3
        assert "type_breakdown" in metadata
        assert metadata["type_breakdown"]["item"] == 2
        assert metadata["type_breakdown"]["location"] == 1

    def test_extract_all_resource_types(self, mock_llm_client: MagicMock) -> None:
        """Test extraction of all resource types."""
        response = {
            "resources": [
                {"name": "bread", "resource_type": "item"},
                {"name": "kitchen", "resource_type": "location"},
                {"name": "order_queue", "resource_type": "queue"},
                {"name": "delivery", "resource_type": "service"},
            ]
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="extract",
            reasoning=json.dumps(response),
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("A bakery with delivery")

        assert len(resources) == 4
        types = [r.resource_type for r in resources]
        assert set(types) == {"item", "location", "queue", "service"}

    def test_extract_uses_action_field_as_fallback(self, mock_llm_client: MagicMock) -> None:
        """Test that action field is used when reasoning is empty."""
        response = {"resources": [{"name": "item", "resource_type": "item"}]}
        mock_llm_client.query.return_value = BehaviorResponse(
            action=json.dumps(response),
            reasoning="",
        )

        extractor = ResourceExtractor(llm_client=mock_llm_client)
        resources = extractor.extract("Test domain")

        assert len(resources) == 1
        assert resources[0].name == "item"

    def test_llm_exception_wrapped(self, mock_llm_client: MagicMock) -> None:
        """Test that LLM exceptions are wrapped in ResourceExtractorError."""
        mock_llm_client.query.side_effect = RuntimeError("API error")

        extractor = ResourceExtractor(llm_client=mock_llm_client)

        with pytest.raises(ResourceExtractorError, match="Failed to extract resources"):
            extractor.extract("Test domain")

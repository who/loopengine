"""Tests for the domains API endpoints."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import SecretStr

from loopengine.api.domains import (
    CreateDomainRequest,
    CreateDomainResponse,
    _generate_domain_id,
    router,
)
from loopengine.behaviors import (
    DomainParser,
    DomainParserError,
    DomainSchema,
    DomainStore,
    DomainStoreError,
)
from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.domain_parser import AgentTypeSchema
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
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    storage = tmp_path / "domains"
    storage.mkdir()
    return storage


@pytest.fixture
def sample_schema() -> DomainSchema:
    """Create a sample domain schema."""
    return DomainSchema(
        domain_type="sandwich shop",
        description="A fast-food sandwich shop",
        agent_types=[
            AgentTypeSchema(
                name="sandwich_maker",
                role="Makes sandwiches",
                capabilities=["make_sandwich", "wrap_sandwich"],
            ),
            AgentTypeSchema(
                name="customer",
                role="Orders food",
                capabilities=["place_order", "pay"],
            ),
        ],
    )


@pytest.fixture
def sample_llm_response() -> dict:
    """Create a sample LLM response for parsing."""
    return {
        "domain_type": "sandwich shop",
        "description": "A fast-food sandwich shop",
        "agent_types": [
            {
                "name": "sandwich_maker",
                "role": "Makes sandwiches",
                "capabilities": ["make_sandwich", "wrap_sandwich"],
            },
            {
                "name": "customer",
                "role": "Orders food",
                "capabilities": ["place_order", "pay"],
            },
        ],
        "resources": [],
        "interactions": [],
    }


@pytest.fixture
def app():
    """Create a FastAPI test app with the domains router."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app) -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestCreateDomainRequest:
    """Tests for the CreateDomainRequest model."""

    def test_valid_request(self) -> None:
        """Test creating a valid request."""
        request = CreateDomainRequest(description="A sandwich shop")
        assert request.description == "A sandwich shop"
        assert request.domain_id is None

    def test_valid_request_with_domain_id(self) -> None:
        """Test creating a request with custom domain_id."""
        request = CreateDomainRequest(description="A sandwich shop", domain_id="my_shop")
        assert request.description == "A sandwich shop"
        assert request.domain_id == "my_shop"

    def test_description_stripped(self) -> None:
        """Test that description is stripped of whitespace."""
        request = CreateDomainRequest(description="  A sandwich shop  ")
        assert request.description == "A sandwich shop"

    def test_empty_description_raises_error(self) -> None:
        """Test that empty description raises validation error."""
        with pytest.raises(ValueError) as exc_info:
            CreateDomainRequest(description="")
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_whitespace_description_raises_error(self) -> None:
        """Test that whitespace-only description raises validation error."""
        with pytest.raises(ValueError) as exc_info:
            CreateDomainRequest(description="   ")
        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()

    def test_invalid_domain_id_raises_error(self) -> None:
        """Test that invalid domain_id raises validation error."""
        with pytest.raises(ValueError) as exc_info:
            CreateDomainRequest(description="A shop", domain_id="invalid/id")
        assert "alphanumeric" in str(exc_info.value).lower()

    def test_domain_id_with_hyphens_allowed(self) -> None:
        """Test that domain_id with hyphens is allowed."""
        request = CreateDomainRequest(description="A shop", domain_id="my-shop-v1")
        assert request.domain_id == "my-shop-v1"

    def test_domain_id_with_underscores_allowed(self) -> None:
        """Test that domain_id with underscores is allowed."""
        request = CreateDomainRequest(description="A shop", domain_id="my_shop_v1")
        assert request.domain_id == "my_shop_v1"

    def test_empty_domain_id_becomes_none(self) -> None:
        """Test that empty domain_id becomes None."""
        request = CreateDomainRequest(description="A shop", domain_id="")
        assert request.domain_id is None

    def test_whitespace_domain_id_becomes_none(self) -> None:
        """Test that whitespace domain_id becomes None."""
        request = CreateDomainRequest(description="A shop", domain_id="   ")
        assert request.domain_id is None


class TestGenerateDomainId:
    """Tests for the _generate_domain_id function."""

    def test_simple_domain_type(self) -> None:
        """Test generating ID from simple domain type."""
        assert _generate_domain_id("shop") == "shop"

    def test_multi_word_domain_type(self) -> None:
        """Test generating ID from multi-word domain type."""
        assert _generate_domain_id("sandwich shop") == "sandwich_shop"

    def test_uppercase_domain_type(self) -> None:
        """Test generating ID from uppercase domain type."""
        assert _generate_domain_id("COFFEE SHOP") == "coffee_shop"

    def test_special_chars_replaced(self) -> None:
        """Test that special characters are replaced."""
        assert _generate_domain_id("flower's shop!") == "flower_s_shop"

    def test_multiple_spaces_consolidated(self) -> None:
        """Test that multiple spaces become single underscore."""
        assert _generate_domain_id("a   big   shop") == "a_big_shop"

    def test_leading_trailing_removed(self) -> None:
        """Test that leading/trailing underscores are removed."""
        assert _generate_domain_id("  shop  ") == "shop"

    def test_empty_becomes_domain(self) -> None:
        """Test that empty string becomes 'domain'."""
        assert _generate_domain_id("") == "domain"

    def test_special_only_becomes_domain(self) -> None:
        """Test that special-chars-only becomes 'domain'."""
        assert _generate_domain_id("!@#$%") == "domain"


class TestCreateDomainEndpoint:
    """Tests for the POST /api/v1/domains endpoint."""

    def test_create_domain_success(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_llm_response: dict,
        mock_llm_client: MagicMock,
        mock_config: LLMConfig,
    ) -> None:
        """Test successful domain creation."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(sample_llm_response),
            metadata={},
        )

        mock_parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        mock_store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_parser", return_value=mock_parser):
            with patch("loopengine.api.domains._get_store", return_value=mock_store):
                response = client.post(
                    "/api/v1/domains",
                    json={"description": "A sandwich shop where employees make sandwiches"},
                )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["domain_id"] == "sandwich_shop"
        assert data["schema"]["domain_type"] == "sandwich shop"
        assert len(data["schema"]["agent_types"]) == 2
        assert data["metadata"]["version"] == 1

    def test_create_domain_with_custom_id(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_llm_response: dict,
        mock_llm_client: MagicMock,
        mock_config: LLMConfig,
    ) -> None:
        """Test domain creation with custom domain_id."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(sample_llm_response),
            metadata={},
        )

        mock_parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        mock_store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_parser", return_value=mock_parser):
            with patch("loopengine.api.domains._get_store", return_value=mock_store):
                response = client.post(
                    "/api/v1/domains",
                    json={
                        "description": "A sandwich shop",
                        "domain_id": "my-custom-shop",
                    },
                )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["domain_id"] == "my-custom-shop"

    def test_update_existing_domain(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_llm_response: dict,
        mock_llm_client: MagicMock,
        mock_config: LLMConfig,
    ) -> None:
        """Test updating an existing domain increments version."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(sample_llm_response),
            metadata={},
        )

        mock_parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        mock_store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_parser", return_value=mock_parser):
            with patch("loopengine.api.domains._get_store", return_value=mock_store):
                # First creation
                response1 = client.post(
                    "/api/v1/domains",
                    json={"description": "A sandwich shop", "domain_id": "test_shop"},
                )
                assert response1.status_code == status.HTTP_201_CREATED
                assert response1.json()["metadata"]["version"] == 1

                # Second update
                response2 = client.post(
                    "/api/v1/domains",
                    json={"description": "A sandwich shop updated", "domain_id": "test_shop"},
                )
                assert response2.status_code == status.HTTP_201_CREATED
                assert response2.json()["metadata"]["version"] == 2

    def test_create_domain_empty_description_returns_400(
        self,
        client: TestClient,
    ) -> None:
        """Test that empty description returns 400 error."""
        response = client.post(
            "/api/v1/domains",
            json={"description": ""},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_domain_missing_description_returns_422(
        self,
        client: TestClient,
    ) -> None:
        """Test that missing description returns 422 error."""
        response = client.post(
            "/api/v1/domains",
            json={},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_domain_parser_error_returns_400(
        self,
        client: TestClient,
        temp_storage_dir: Path,
    ) -> None:
        """Test that parser error returns 400 error."""
        mock_parser = MagicMock(spec=DomainParser)
        mock_parser.parse_with_metadata.side_effect = DomainParserError("Parse failed")
        mock_store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_parser", return_value=mock_parser):
            with patch("loopengine.api.domains._get_store", return_value=mock_store):
                response = client.post(
                    "/api/v1/domains",
                    json={"description": "A test domain"},
                )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Failed to parse" in response.json()["detail"]

    def test_create_domain_storage_error_returns_500(
        self,
        client: TestClient,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that storage error returns 500 error."""
        mock_parser = MagicMock(spec=DomainParser)
        mock_parser.parse_with_metadata.return_value = (sample_schema, {"latency_ms": 100})

        mock_store = MagicMock(spec=DomainStore)
        mock_store.save.side_effect = DomainStoreError("Storage failed")

        with patch("loopengine.api.domains._get_parser", return_value=mock_parser):
            with patch("loopengine.api.domains._get_store", return_value=mock_store):
                response = client.post(
                    "/api/v1/domains",
                    json={"description": "A test domain"},
                )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to store" in response.json()["detail"]

    def test_create_domain_internal_error_returns_500(
        self,
        client: TestClient,
    ) -> None:
        """Test that unexpected error returns 500 error."""
        mock_parser = MagicMock(spec=DomainParser)
        mock_parser.parse_with_metadata.side_effect = RuntimeError("Unexpected error")

        with patch("loopengine.api.domains._get_parser", return_value=mock_parser):
            response = client.post(
                "/api/v1/domains",
                json={"description": "A test domain"},
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Internal error" in response.json()["detail"]

    def test_create_domain_response_includes_metadata(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_llm_response: dict,
        mock_llm_client: MagicMock,
        mock_config: LLMConfig,
    ) -> None:
        """Test that response includes all metadata fields."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(sample_llm_response),
            metadata={},
        )

        mock_parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        mock_store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_parser", return_value=mock_parser):
            with patch("loopengine.api.domains._get_store", return_value=mock_store):
                response = client.post(
                    "/api/v1/domains",
                    json={"description": "A sandwich shop"},
                )

        assert response.status_code == status.HTTP_201_CREATED
        metadata = response.json()["metadata"]
        assert "domain_id" in metadata
        assert "version" in metadata
        assert "created_at" in metadata
        assert "updated_at" in metadata
        assert "parse_latency_ms" in metadata
        assert "agent_types_extracted" in metadata
        assert "resources_extracted" in metadata
        assert "interactions_extracted" in metadata

    def test_domain_persisted_to_storage(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_llm_response: dict,
        mock_llm_client: MagicMock,
        mock_config: LLMConfig,
    ) -> None:
        """Test that domain is persisted to storage."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(sample_llm_response),
            metadata={},
        )

        mock_parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        mock_store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_parser", return_value=mock_parser):
            with patch("loopengine.api.domains._get_store", return_value=mock_store):
                response = client.post(
                    "/api/v1/domains",
                    json={"description": "A sandwich shop", "domain_id": "persisted_shop"},
                )

        assert response.status_code == status.HTTP_201_CREATED

        # Verify domain was persisted
        stored = mock_store.load("persisted_shop")
        assert stored.schema_.domain_type == "sandwich shop"
        assert stored.metadata.version == 1


class TestCreateDomainResponse:
    """Tests for the CreateDomainResponse model."""

    def test_response_serialization(self, sample_schema: DomainSchema) -> None:
        """Test response serializes correctly."""
        response = CreateDomainResponse(
            domain_id="test",
            schema=sample_schema,
            metadata={"version": 1, "created_at": "2024-01-01T00:00:00Z"},
        )

        data = response.model_dump(by_alias=True)
        assert "schema" in data
        assert "schema_" not in data
        assert data["domain_id"] == "test"
        assert data["schema"]["domain_type"] == "sandwich shop"


class TestRouterIntegration:
    """Integration tests for the domains router."""

    def test_router_has_correct_prefix(self) -> None:
        """Test router has correct prefix."""
        assert router.prefix == "/api/v1/domains"

    def test_router_has_correct_tags(self) -> None:
        """Test router has correct tags."""
        assert "domains" in router.tags

    def test_post_endpoint_exists(self) -> None:
        """Test POST endpoint is registered."""
        # Route path includes prefix since it's the root path
        routes = [r for r in router.routes if r.path == "/api/v1/domains"]
        assert len(routes) == 1
        assert "POST" in routes[0].methods

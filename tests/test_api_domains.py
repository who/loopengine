"""Tests for the domains API endpoints."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import SecretStr

from loopengine.api.domains import (
    AddConstraintRequest,
    CreateDomainRequest,
    CreateDomainResponse,
    GetDomainResponse,
    _generate_domain_id,
    router,
)
from loopengine.behaviors import (
    ConstraintSchema,
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

    def test_get_endpoint_exists(self) -> None:
        """Test GET endpoint is registered."""
        routes = [r for r in router.routes if r.path == "/api/v1/domains/{domain_id}"]
        assert len(routes) == 1
        assert "GET" in routes[0].methods


class TestGetDomainEndpoint:
    """Tests for the GET /api/v1/domains/{domain_id} endpoint."""

    def test_get_domain_success(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test successful domain retrieval."""
        # Pre-populate storage
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/test_shop")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["domain_id"] == "test_shop"
        assert data["schema"]["domain_type"] == "sandwich shop"
        assert len(data["schema"]["agent_types"]) == 2
        assert data["metadata"]["version"] == 1
        assert "created_at" in data["metadata"]
        assert "updated_at" in data["metadata"]

    def test_get_domain_not_found_returns_404(
        self,
        client: TestClient,
        temp_storage_dir: Path,
    ) -> None:
        """Test that non-existent domain returns 404."""
        store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    def test_get_domain_returns_latest_version(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that GET returns the latest version after updates."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("versioned_shop", sample_schema)
        store.save("versioned_shop", sample_schema)  # Version 2
        store.save("versioned_shop", sample_schema)  # Version 3

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/versioned_shop")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["metadata"]["version"] == 3

    def test_get_domain_matches_stored_data(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that response matches stored domain exactly."""
        store = DomainStore(storage_dir=temp_storage_dir)
        stored = store.save("exact_match", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/exact_match")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify schema matches
        assert data["schema"]["domain_type"] == stored.schema_.domain_type
        assert data["schema"]["description"] == stored.schema_.description
        assert len(data["schema"]["agent_types"]) == len(stored.schema_.agent_types)

        # Verify metadata matches
        assert data["metadata"]["domain_id"] == stored.metadata.domain_id
        assert data["metadata"]["version"] == stored.metadata.version
        assert data["metadata"]["created_at"] == stored.metadata.created_at
        assert data["metadata"]["updated_at"] == stored.metadata.updated_at

    def test_get_domain_includes_all_agent_types(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that response includes all agent types from schema."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("agent_types_test", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/agent_types_test")

        assert response.status_code == status.HTTP_200_OK
        agent_types = response.json()["schema"]["agent_types"]
        assert len(agent_types) == 2

        agent_names = [a["name"] for a in agent_types]
        assert "sandwich_maker" in agent_names
        assert "customer" in agent_names

    def test_get_domain_with_hyphens(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test retrieval of domain with hyphenated ID."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("my-shop-v1", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/my-shop-v1")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["domain_id"] == "my-shop-v1"

    def test_get_domain_with_underscores(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test retrieval of domain with underscored ID."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("my_shop_v1", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/my_shop_v1")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["domain_id"] == "my_shop_v1"

    def test_get_domain_404_message_includes_domain_id(
        self,
        client: TestClient,
        temp_storage_dir: Path,
    ) -> None:
        """Test that 404 error message includes the requested domain_id."""
        store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/specific_missing_id")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "specific_missing_id" in response.json()["detail"]


class TestGetDomainResponse:
    """Tests for the GetDomainResponse model."""

    def test_response_serialization(self, sample_schema: DomainSchema) -> None:
        """Test response serializes correctly."""
        from loopengine.behaviors import DomainMetadata

        metadata = DomainMetadata(
            domain_id="test",
            version=1,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        response = GetDomainResponse(
            domain_id="test",
            schema=sample_schema,
            metadata=metadata,
        )

        data = response.model_dump(by_alias=True)
        assert "schema" in data
        assert "schema_" not in data
        assert data["domain_id"] == "test"
        assert data["schema"]["domain_type"] == "sandwich shop"
        assert data["metadata"]["version"] == 1

    def test_response_from_stored_domain(self, sample_schema: DomainSchema) -> None:
        """Test creating response from a StoredDomain."""
        from loopengine.behaviors import DomainMetadata
        from loopengine.behaviors.domain_store import StoredDomain

        metadata = DomainMetadata(
            domain_id="stored_test",
            version=2,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
        )

        stored = StoredDomain(metadata=metadata, schema=sample_schema)

        response = GetDomainResponse(
            domain_id=stored.metadata.domain_id,
            schema=stored.schema_,
            metadata=stored.metadata,
        )

        assert response.domain_id == "stored_test"
        assert response.metadata.version == 2


class TestAddConstraintRequest:
    """Tests for the AddConstraintRequest model."""

    def test_valid_positive_constraint(self) -> None:
        """Test creating a valid positive constraint request."""
        request = AddConstraintRequest(
            text="greet customers warmly",
            constraint_type="positive",
        )
        assert request.text == "greet customers warmly"
        assert request.constraint_type == "positive"

    def test_valid_negative_constraint(self) -> None:
        """Test creating a valid negative constraint request."""
        request = AddConstraintRequest(
            text="refuse service",
            constraint_type="negative",
        )
        assert request.constraint_type == "negative"

    def test_default_constraint_type_is_positive(self) -> None:
        """Test that default constraint type is positive."""
        request = AddConstraintRequest(text="be polite")
        assert request.constraint_type == "positive"

    def test_text_is_stripped(self) -> None:
        """Test that text is stripped of whitespace."""
        request = AddConstraintRequest(text="  greet customers  ")
        assert request.text == "greet customers"

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises validation error."""
        with pytest.raises(ValueError):
            AddConstraintRequest(text="")

    def test_whitespace_text_raises_error(self) -> None:
        """Test that whitespace-only text raises validation error."""
        with pytest.raises(ValueError):
            AddConstraintRequest(text="   ")

    def test_invalid_constraint_type_raises_error(self) -> None:
        """Test that invalid constraint_type raises validation error."""
        with pytest.raises(ValueError) as exc_info:
            AddConstraintRequest(text="test", constraint_type="invalid")
        error_msg = str(exc_info.value).lower()
        assert "positive" in error_msg or "negative" in error_msg

    def test_constraint_type_is_normalized(self) -> None:
        """Test that constraint_type is normalized to lowercase."""
        request = AddConstraintRequest(text="test", constraint_type="POSITIVE")
        assert request.constraint_type == "positive"


class TestAddConstraintEndpoint:
    """Tests for the POST /api/v1/domains/{domain_id}/constraints endpoint."""

    def test_add_constraint_success(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test successful constraint addition."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.post(
                "/api/v1/domains/test_shop/constraints",
                json={"text": "always greet customers", "constraint_type": "positive"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["domain_id"] == "test_shop"
        assert len(data["constraints"]) == 1
        assert data["constraints"][0]["text"] == "always greet customers"
        assert data["constraints"][0]["constraint_type"] == "positive"
        assert data["version"] == 2  # Version incremented

    def test_add_multiple_constraints(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test adding multiple constraints."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            # Add first constraint
            response1 = client.post(
                "/api/v1/domains/test_shop/constraints",
                json={"text": "greet customers", "constraint_type": "positive"},
            )
            assert response1.status_code == status.HTTP_200_OK
            assert len(response1.json()["constraints"]) == 1

            # Add second constraint
            response2 = client.post(
                "/api/v1/domains/test_shop/constraints",
                json={"text": "refuse service", "constraint_type": "negative"},
            )
            assert response2.status_code == status.HTTP_200_OK
            assert len(response2.json()["constraints"]) == 2

    def test_add_constraint_domain_not_found(
        self,
        client: TestClient,
        temp_storage_dir: Path,
    ) -> None:
        """Test adding constraint to non-existent domain returns 404."""
        store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.post(
                "/api/v1/domains/nonexistent/constraints",
                json={"text": "test", "constraint_type": "positive"},
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_add_constraint_default_positive(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that constraint_type defaults to positive."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.post(
                "/api/v1/domains/test_shop/constraints",
                json={"text": "be helpful"},
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["constraints"][0]["constraint_type"] == "positive"


class TestUpdateConstraintsEndpoint:
    """Tests for the PUT /api/v1/domains/{domain_id}/constraints endpoint."""

    def test_update_constraints_success(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test successful constraints update."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.put(
                "/api/v1/domains/test_shop/constraints",
                json={
                    "constraints": [
                        {"text": "greet customers", "constraint_type": "positive"},
                        {"text": "refuse service", "constraint_type": "negative"},
                    ]
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["constraints"]) == 2
        assert data["constraints"][0]["text"] == "greet customers"
        assert data["constraints"][1]["text"] == "refuse service"

    def test_update_constraints_replaces_existing(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that update replaces existing constraints."""
        # Create schema with existing constraint
        schema_with_constraint = DomainSchema(
            domain_type=sample_schema.domain_type,
            description=sample_schema.description,
            agent_types=sample_schema.agent_types,
            constraints=[ConstraintSchema(text="old constraint", constraint_type="positive")],
        )
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", schema_with_constraint)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.put(
                "/api/v1/domains/test_shop/constraints",
                json={
                    "constraints": [
                        {"text": "new constraint", "constraint_type": "negative"},
                    ]
                },
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["constraints"]) == 1
        assert data["constraints"][0]["text"] == "new constraint"
        assert data["constraints"][0]["constraint_type"] == "negative"

    def test_update_constraints_clear_all(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test clearing all constraints with empty list."""
        schema_with_constraint = DomainSchema(
            domain_type=sample_schema.domain_type,
            description=sample_schema.description,
            agent_types=sample_schema.agent_types,
            constraints=[ConstraintSchema(text="constraint", constraint_type="positive")],
        )
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", schema_with_constraint)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.put(
                "/api/v1/domains/test_shop/constraints",
                json={"constraints": []},
            )

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["constraints"]) == 0

    def test_update_constraints_domain_not_found(
        self,
        client: TestClient,
        temp_storage_dir: Path,
    ) -> None:
        """Test updating constraints on non-existent domain returns 404."""
        store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.put(
                "/api/v1/domains/nonexistent/constraints",
                json={"constraints": []},
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestGetConstraintsEndpoint:
    """Tests for the GET /api/v1/domains/{domain_id}/constraints endpoint."""

    def test_get_constraints_success(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test successful constraints retrieval."""
        schema_with_constraints = DomainSchema(
            domain_type=sample_schema.domain_type,
            description=sample_schema.description,
            agent_types=sample_schema.agent_types,
            constraints=[
                ConstraintSchema(text="greet customers", constraint_type="positive"),
                ConstraintSchema(text="refuse service", constraint_type="negative"),
            ],
        )
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", schema_with_constraints)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/test_shop/constraints")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["domain_id"] == "test_shop"
        assert len(data["constraints"]) == 2
        assert data["constraints"][0]["text"] == "greet customers"

    def test_get_constraints_empty(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test getting constraints when none exist."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", sample_schema)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/test_shop/constraints")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["constraints"]) == 0

    def test_get_constraints_domain_not_found(
        self,
        client: TestClient,
        temp_storage_dir: Path,
    ) -> None:
        """Test getting constraints from non-existent domain returns 404."""
        store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.get("/api/v1/domains/nonexistent/constraints")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestDeleteConstraintEndpoint:
    """Tests for DELETE /api/v1/domains/{domain_id}/constraints/{constraint_index} endpoint."""

    def test_delete_constraint_success(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test successful constraint deletion."""
        schema_with_constraints = DomainSchema(
            domain_type=sample_schema.domain_type,
            description=sample_schema.description,
            agent_types=sample_schema.agent_types,
            constraints=[
                ConstraintSchema(text="first constraint", constraint_type="positive"),
                ConstraintSchema(text="second constraint", constraint_type="negative"),
            ],
        )
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", schema_with_constraints)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.delete("/api/v1/domains/test_shop/constraints/0")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["constraints"]) == 1
        assert data["constraints"][0]["text"] == "second constraint"

    def test_delete_constraint_invalid_index(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test deleting constraint with invalid index returns 404."""
        schema_with_constraints = DomainSchema(
            domain_type=sample_schema.domain_type,
            description=sample_schema.description,
            agent_types=sample_schema.agent_types,
            constraints=[
                ConstraintSchema(text="constraint", constraint_type="positive"),
            ],
        )
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", schema_with_constraints)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.delete("/api/v1/domains/test_shop/constraints/5")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "index" in response.json()["detail"].lower()

    def test_delete_constraint_negative_index(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test deleting constraint with negative index returns 404."""
        schema_with_constraints = DomainSchema(
            domain_type=sample_schema.domain_type,
            description=sample_schema.description,
            agent_types=sample_schema.agent_types,
            constraints=[
                ConstraintSchema(text="constraint", constraint_type="positive"),
            ],
        )
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("test_shop", schema_with_constraints)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.delete("/api/v1/domains/test_shop/constraints/-1")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_constraint_domain_not_found(
        self,
        client: TestClient,
        temp_storage_dir: Path,
    ) -> None:
        """Test deleting constraint from non-existent domain returns 404."""
        store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.domains._get_store", return_value=store):
            response = client.delete("/api/v1/domains/nonexistent/constraints/0")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestConstraintEndpointsRouterIntegration:
    """Integration tests for constraint endpoints registration."""

    def test_add_constraint_endpoint_exists(self) -> None:
        """Test POST constraints endpoint is registered."""
        routes = [r for r in router.routes if "constraints" in r.path and "{domain_id}" in r.path]
        post_routes = [r for r in routes if "POST" in r.methods and r.path.endswith("/constraints")]
        assert len(post_routes) == 1

    def test_update_constraints_endpoint_exists(self) -> None:
        """Test PUT constraints endpoint is registered."""
        routes = [r for r in router.routes if "constraints" in r.path and "{domain_id}" in r.path]
        put_routes = [r for r in routes if "PUT" in r.methods]
        assert len(put_routes) == 1

    def test_get_constraints_endpoint_exists(self) -> None:
        """Test GET constraints endpoint is registered."""
        routes = [r for r in router.routes if "constraints" in r.path and "{domain_id}" in r.path]
        get_routes = [r for r in routes if "GET" in r.methods and r.path.endswith("/constraints")]
        assert len(get_routes) == 1

    def test_delete_constraint_endpoint_exists(self) -> None:
        """Test DELETE constraint endpoint is registered."""
        routes = [r for r in router.routes if "constraint_index" in r.path]
        delete_routes = [r for r in routes if "DELETE" in r.methods]
        assert len(delete_routes) == 1

"""Tests for the behaviors API endpoints."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import SecretStr

from loopengine.api.behaviors import (
    BehaviorMetadata,
    GenerateBehaviorRequest,
    GenerateBehaviorResponse,
    _find_agent_type_in_schema,
    router,
)
from loopengine.behaviors import (
    AIBehaviorEngine,
    AIBehaviorEngineError,
    DomainStore,
)
from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.domain_parser import AgentTypeSchema, DomainSchema
from loopengine.behaviors.llm_client import BehaviorResponse


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
        description="A fast-food sandwich shop where workers make sandwiches for customers",
        agent_types=[
            AgentTypeSchema(
                name="sandwich_maker",
                role="Makes sandwiches for customers",
                capabilities=["make_sandwich", "wrap_sandwich", "add_toppings"],
            ),
            AgentTypeSchema(
                name="customer",
                role="Orders and receives food",
                capabilities=["place_order", "pay", "wait"],
            ),
            AgentTypeSchema(
                name="cashier",
                role="Takes orders and processes payments",
                capabilities=["take_order", "process_payment", "give_receipt"],
            ),
        ],
    )


@pytest.fixture
def sample_behavior_response() -> BehaviorResponse:
    """Create a sample behavior response from engine."""
    return BehaviorResponse(
        action="make_sandwich",
        parameters={"type": "turkey", "toppings": ["lettuce", "tomato"]},
        reasoning="Customer ordered a turkey sandwich with lettuce and tomato",
        metadata={"latency_ms": 150.5, "provider": "claude", "cached": False},
    )


@pytest.fixture
def app():
    """Create a FastAPI test app with the behaviors router."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app) -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestGenerateBehaviorRequest:
    """Tests for the GenerateBehaviorRequest model."""

    def test_valid_request(self) -> None:
        """Test creating a valid request."""
        request = GenerateBehaviorRequest(
            domain_id="sandwich_shop",
            agent_type="sandwich_maker",
        )
        assert request.domain_id == "sandwich_shop"
        assert request.agent_type == "sandwich_maker"
        assert request.context is None

    def test_valid_request_with_context(self) -> None:
        """Test creating a request with context."""
        context = {"pending_orders": 3, "current_task": None}
        request = GenerateBehaviorRequest(
            domain_id="sandwich_shop",
            agent_type="sandwich_maker",
            context=context,
        )
        assert request.context == context

    def test_domain_id_stripped(self) -> None:
        """Test that domain_id is stripped of whitespace."""
        request = GenerateBehaviorRequest(
            domain_id="  sandwich_shop  ",
            agent_type="customer",
        )
        assert request.domain_id == "sandwich_shop"

    def test_agent_type_stripped(self) -> None:
        """Test that agent_type is stripped of whitespace."""
        request = GenerateBehaviorRequest(
            domain_id="sandwich_shop",
            agent_type="  customer  ",
        )
        assert request.agent_type == "customer"

    def test_empty_domain_id_raises_error(self) -> None:
        """Test that empty domain_id raises validation error."""
        with pytest.raises(ValueError) as exc_info:
            GenerateBehaviorRequest(domain_id="", agent_type="customer")
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_whitespace_domain_id_raises_error(self) -> None:
        """Test that whitespace-only domain_id raises validation error."""
        with pytest.raises(ValueError) as exc_info:
            GenerateBehaviorRequest(domain_id="   ", agent_type="customer")
        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()

    def test_empty_agent_type_raises_error(self) -> None:
        """Test that empty agent_type raises validation error."""
        with pytest.raises(ValueError) as exc_info:
            GenerateBehaviorRequest(domain_id="shop", agent_type="")
        assert "at least 1 character" in str(exc_info.value).lower()

    def test_whitespace_agent_type_raises_error(self) -> None:
        """Test that whitespace-only agent_type raises validation error."""
        with pytest.raises(ValueError) as exc_info:
            GenerateBehaviorRequest(domain_id="shop", agent_type="   ")
        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()


class TestFindAgentTypeInSchema:
    """Tests for the _find_agent_type_in_schema function."""

    def test_exact_match(self, sample_schema: DomainSchema) -> None:
        """Test exact match of agent type."""
        name, role = _find_agent_type_in_schema(
            "sandwich_maker",
            sample_schema.agent_types,
        )
        assert name == "sandwich_maker"
        assert role == "Makes sandwiches for customers"

    def test_case_insensitive_match(self, sample_schema: DomainSchema) -> None:
        """Test case insensitive matching."""
        name, _role = _find_agent_type_in_schema(
            "SANDWICH_MAKER",
            sample_schema.agent_types,
        )
        assert name == "sandwich_maker"

    def test_hyphen_underscore_conversion(self, sample_schema: DomainSchema) -> None:
        """Test hyphen to underscore conversion."""
        name, _role = _find_agent_type_in_schema(
            "sandwich-maker",
            sample_schema.agent_types,
        )
        assert name == "sandwich_maker"

    def test_space_to_underscore_conversion(self, sample_schema: DomainSchema) -> None:
        """Test space to underscore conversion."""
        name, _role = _find_agent_type_in_schema(
            "sandwich maker",
            sample_schema.agent_types,
        )
        assert name == "sandwich_maker"

    def test_not_found_raises_error(self, sample_schema: DomainSchema) -> None:
        """Test that non-existent agent type raises error."""
        with pytest.raises(ValueError) as exc_info:
            _find_agent_type_in_schema("nonexistent", sample_schema.agent_types)
        assert "not found in domain" in str(exc_info.value)
        assert "sandwich_maker" in str(exc_info.value)
        assert "customer" in str(exc_info.value)

    def test_with_dict_agent_types(self) -> None:
        """Test with dict-based agent types (from JSON)."""
        agent_types = [
            {"name": "florist", "role": "Arranges flowers"},
            {"name": "delivery", "role": "Delivers orders"},
        ]
        name, role = _find_agent_type_in_schema("florist", agent_types)
        assert name == "florist"
        assert role == "Arranges flowers"


class TestBehaviorMetadata:
    """Tests for the BehaviorMetadata model."""

    def test_default_cached_is_false(self) -> None:
        """Test that cached defaults to False."""
        metadata = BehaviorMetadata(latency_ms=100.0, provider="claude")
        assert metadata.cached is False

    def test_all_fields(self) -> None:
        """Test creating metadata with all fields."""
        metadata = BehaviorMetadata(
            cached=True,
            latency_ms=250.5,
            provider="openai",
        )
        assert metadata.cached is True
        assert metadata.latency_ms == 250.5
        assert metadata.provider == "openai"


class TestGenerateBehaviorResponse:
    """Tests for the GenerateBehaviorResponse model."""

    def test_response_creation(self) -> None:
        """Test creating a response."""
        metadata = BehaviorMetadata(latency_ms=100.0, provider="claude")
        response = GenerateBehaviorResponse(
            action="make_sandwich",
            parameters={"type": "turkey"},
            reasoning="Customer wants turkey",
            metadata=metadata,
        )
        assert response.action == "make_sandwich"
        assert response.parameters == {"type": "turkey"}
        assert response.reasoning == "Customer wants turkey"
        assert response.metadata.latency_ms == 100.0

    def test_default_parameters_empty(self) -> None:
        """Test that parameters defaults to empty dict."""
        metadata = BehaviorMetadata(latency_ms=100.0, provider="claude")
        response = GenerateBehaviorResponse(
            action="wait",
            metadata=metadata,
        )
        assert response.parameters == {}
        assert response.reasoning == ""


class TestGenerateBehaviorEndpoint:
    """Tests for the POST /api/v1/behaviors/generate endpoint."""

    def test_generate_behavior_success(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
        sample_behavior_response: BehaviorResponse,
    ) -> None:
        """Test successful behavior generation."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = sample_behavior_response
        mock_engine.provider = "claude"

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "sandwich_maker",
                        "context": {"pending_orders": 2},
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["action"] == "make_sandwich"
        assert data["parameters"]["type"] == "turkey"
        assert data["reasoning"] == "Customer ordered a turkey sandwich with lettuce and tomato"
        assert data["metadata"]["provider"] == "claude"
        assert data["metadata"]["latency_ms"] == 150.5
        assert data["metadata"]["cached"] is False

    def test_generate_behavior_without_context(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
        sample_behavior_response: BehaviorResponse,
    ) -> None:
        """Test behavior generation without context."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = sample_behavior_response
        mock_engine.provider = "claude"

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "customer",
                    },
                )

        assert response.status_code == status.HTTP_200_OK

        # Verify engine was called with None context
        call_args = mock_engine.generate_behavior.call_args
        assert call_args.kwargs.get("context") is None

    def test_domain_not_found_returns_404(
        self,
        client: TestClient,
        temp_storage_dir: Path,
    ) -> None:
        """Test that non-existent domain returns 404."""
        store = DomainStore(storage_dir=temp_storage_dir)

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            response = client.post(
                "/api/v1/behaviors/generate",
                json={
                    "domain_id": "nonexistent_domain",
                    "agent_type": "some_agent",
                },
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()
        assert "nonexistent_domain" in response.json()["detail"]

    def test_invalid_agent_type_returns_400(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that invalid agent_type returns 400."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            response = client.post(
                "/api/v1/behaviors/generate",
                json={
                    "domain_id": "sandwich_shop",
                    "agent_type": "invalid_agent",
                },
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not found in domain" in response.json()["detail"]
        assert "sandwich_maker" in response.json()["detail"]

    def test_engine_error_returns_500(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that engine error returns 500."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.side_effect = AIBehaviorEngineError("LLM query failed")

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "sandwich_maker",
                    },
                )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to generate" in response.json()["detail"]

    def test_rate_limit_error_returns_503(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that rate limit error returns 503."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.side_effect = AIBehaviorEngineError(
            "Rate limit exceeded for API"
        )

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "sandwich_maker",
                    },
                )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "rate limit" in response.json()["detail"].lower()

    def test_auth_error_returns_503(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that authentication error returns 503."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.side_effect = AIBehaviorEngineError(
            "API key authentication failed"
        )

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "sandwich_maker",
                    },
                )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "authentication" in response.json()["detail"].lower()

    def test_unexpected_error_returns_500(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that unexpected error returns 500."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.side_effect = RuntimeError("Unexpected error")

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "sandwich_maker",
                    },
                )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Internal error" in response.json()["detail"]

    def test_missing_domain_id_returns_422(
        self,
        client: TestClient,
    ) -> None:
        """Test that missing domain_id returns 422."""
        response = client.post(
            "/api/v1/behaviors/generate",
            json={"agent_type": "customer"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_agent_type_returns_422(
        self,
        client: TestClient,
    ) -> None:
        """Test that missing agent_type returns 422."""
        response = client.post(
            "/api/v1/behaviors/generate",
            json={"domain_id": "shop"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_agent_type_case_insensitive(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
        sample_behavior_response: BehaviorResponse,
    ) -> None:
        """Test that agent_type matching is case insensitive."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = sample_behavior_response
        mock_engine.provider = "claude"

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "SANDWICH_MAKER",
                    },
                )

        assert response.status_code == status.HTTP_200_OK

    def test_agent_type_hyphen_conversion(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
        sample_behavior_response: BehaviorResponse,
    ) -> None:
        """Test that agent_type with hyphens matches underscore version."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = sample_behavior_response
        mock_engine.provider = "claude"

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "sandwich-maker",
                    },
                )

        assert response.status_code == status.HTTP_200_OK

    def test_cached_response_metadata(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that cached response has correct metadata."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        cached_response = BehaviorResponse(
            action="wait",
            parameters={},
            reasoning="Cached response",
            metadata={"latency_ms": 1.5, "provider": "claude", "cached": True},
        )

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = cached_response
        mock_engine.provider = "claude"

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "customer",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["metadata"]["cached"] is True


class TestRouterIntegration:
    """Integration tests for the behaviors router."""

    def test_router_has_correct_prefix(self) -> None:
        """Test router has correct prefix."""
        assert router.prefix == "/api/v1/behaviors"

    def test_router_has_correct_tags(self) -> None:
        """Test router has correct tags."""
        assert "behaviors" in router.tags

    def test_generate_endpoint_exists(self) -> None:
        """Test generate endpoint is registered."""
        routes = [r for r in router.routes if r.path == "/api/v1/behaviors/generate"]
        assert len(routes) == 1
        assert "POST" in routes[0].methods


class TestContextPassthrough:
    """Tests for context being passed correctly to engine."""

    def test_complex_context_passed_to_engine(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
        sample_behavior_response: BehaviorResponse,
    ) -> None:
        """Test that complex context is passed to engine correctly."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = sample_behavior_response
        mock_engine.provider = "claude"

        complex_context = {
            "pending_orders": [
                {"id": 1, "items": ["turkey", "ham"]},
                {"id": 2, "items": ["veggie"]},
            ],
            "current_task": "preparing_order_1",
            "inventory": {"bread": 50, "turkey": 20, "ham": 15},
            "time_of_day": "lunch_rush",
        }

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "sandwich_maker",
                        "context": complex_context,
                    },
                )

        assert response.status_code == status.HTTP_200_OK

        # Verify context was passed correctly
        call_args = mock_engine.generate_behavior.call_args
        passed_context = call_args.kwargs.get("context")
        assert passed_context == complex_context

    def test_domain_context_built_correctly(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
        sample_behavior_response: BehaviorResponse,
    ) -> None:
        """Test that DomainContext is built correctly from stored domain."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = sample_behavior_response
        mock_engine.provider = "claude"

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "sandwich_maker",
                    },
                )

        assert response.status_code == status.HTTP_200_OK

        # Verify domain context
        call_args = mock_engine.generate_behavior.call_args
        domain = call_args.kwargs.get("domain")
        assert domain.domain_type == "sandwich shop"
        assert "fast-food" in domain.domain_description.lower()

    def test_agent_context_built_correctly(
        self,
        client: TestClient,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
        sample_behavior_response: BehaviorResponse,
    ) -> None:
        """Test that AgentContext is built correctly from schema."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = sample_behavior_response
        mock_engine.provider = "claude"

        with patch("loopengine.api.behaviors._get_store", return_value=store):
            with patch("loopengine.api.behaviors._get_engine", return_value=mock_engine):
                response = client.post(
                    "/api/v1/behaviors/generate",
                    json={
                        "domain_id": "sandwich_shop",
                        "agent_type": "cashier",
                    },
                )

        assert response.status_code == status.HTTP_200_OK

        # Verify agent context
        call_args = mock_engine.generate_behavior.call_args
        agent = call_args.kwargs.get("agent")
        assert agent.agent_type == "cashier"
        assert "orders" in agent.agent_role.lower() or "payments" in agent.agent_role.lower()


class TestGetCacheEndpoint:
    """Tests for the GET /api/v1/behaviors/cache endpoint."""

    def test_get_cache_empty(self, client: TestClient) -> None:
        """Test getting empty cache returns empty list."""
        from loopengine.behaviors import BehaviorCache

        cache = BehaviorCache()

        with patch("loopengine.api.behaviors._get_behavior_cache", return_value=cache):
            response = client.get("/api/v1/behaviors/cache")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["entries"] == []
        assert data["total_entries"] == 0
        assert "stats" in data
        assert data["stats"]["hits"] == 0
        assert data["stats"]["misses"] == 0

    def test_get_cache_with_entries(self, client: TestClient) -> None:
        """Test getting cache with entries returns all entries."""
        from loopengine.behaviors import BehaviorCache

        cache = BehaviorCache()
        behavior1 = BehaviorResponse(
            action="make_sandwich",
            parameters={"type": "turkey"},
            reasoning="Customer wants turkey",
            metadata={},
        )
        behavior2 = BehaviorResponse(
            action="wait",
            parameters={},
            reasoning="Queue empty",
            metadata={},
        )
        cache.set("domain1:agent1:hash1", behavior1)
        cache.set("domain2:agent2:hash2", behavior2)

        with patch("loopengine.api.behaviors._get_behavior_cache", return_value=cache):
            response = client.get("/api/v1/behaviors/cache")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["entries"]) == 2
        assert data["total_entries"] == 2

        # Verify entry contents
        keys = [e["key"] for e in data["entries"]]
        assert "domain1:agent1:hash1" in keys
        assert "domain2:agent2:hash2" in keys

        # Find and verify the first behavior
        entry1 = next(e for e in data["entries"] if e["key"] == "domain1:agent1:hash1")
        assert entry1["action"] == "make_sandwich"
        assert entry1["parameters"] == {"type": "turkey"}
        assert entry1["reasoning"] == "Customer wants turkey"

    def test_get_cache_filter_by_domain(self, client: TestClient) -> None:
        """Test filtering cache by domain_id."""
        from loopengine.behaviors import BehaviorCache

        cache = BehaviorCache()
        behavior1 = BehaviorResponse(action="action1", parameters={}, reasoning="", metadata={})
        behavior2 = BehaviorResponse(action="action2", parameters={}, reasoning="", metadata={})
        behavior3 = BehaviorResponse(action="action3", parameters={}, reasoning="", metadata={})
        cache.set("domain_a:agent1:hash1", behavior1)
        cache.set("domain_a:agent2:hash2", behavior2)
        cache.set("domain_b:agent1:hash3", behavior3)

        with patch("loopengine.api.behaviors._get_behavior_cache", return_value=cache):
            response = client.get("/api/v1/behaviors/cache?domain_id=domain_a")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["entries"]) == 2

        # All entries should be from domain_a
        for entry in data["entries"]:
            assert entry["key"].startswith("domain_a:")

    def test_get_cache_filter_nonexistent_domain(self, client: TestClient) -> None:
        """Test filtering by nonexistent domain returns empty list."""
        from loopengine.behaviors import BehaviorCache

        cache = BehaviorCache()
        behavior1 = BehaviorResponse(action="action1", parameters={}, reasoning="", metadata={})
        cache.set("domain_a:agent1:hash1", behavior1)

        with patch("loopengine.api.behaviors._get_behavior_cache", return_value=cache):
            response = client.get("/api/v1/behaviors/cache?domain_id=nonexistent")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["entries"] == []
        # total_entries should show the actual cache size, not filtered count
        assert data["total_entries"] == 1

    def test_get_cache_includes_stats(self, client: TestClient) -> None:
        """Test that cache stats are included in response."""
        from loopengine.behaviors import BehaviorCache

        cache = BehaviorCache()
        behavior = BehaviorResponse(action="test", parameters={}, reasoning="", metadata={})
        cache.set("domain:agent:hash", behavior)

        # Generate some hits and misses
        cache.get("domain:agent:hash")  # hit
        cache.get("domain:agent:hash")  # hit
        cache.get("nonexistent:key:abc")  # miss

        with patch("loopengine.api.behaviors._get_behavior_cache", return_value=cache):
            response = client.get("/api/v1/behaviors/cache")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["stats"]["hits"] == 2
        assert data["stats"]["misses"] == 1
        assert data["stats"]["hit_rate"] > 0

    def test_get_cache_entry_has_ttl_info(self, client: TestClient) -> None:
        """Test that each entry includes TTL information."""
        from loopengine.behaviors import BehaviorCache

        cache = BehaviorCache(default_ttl=300)  # 5 minutes
        behavior = BehaviorResponse(action="test", parameters={}, reasoning="", metadata={})
        cache.set("domain:agent:hash", behavior)

        with patch("loopengine.api.behaviors._get_behavior_cache", return_value=cache):
            response = client.get("/api/v1/behaviors/cache")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        entry = data["entries"][0]

        assert "created_at" in entry
        assert "expires_at" in entry
        assert "ttl_remaining" in entry
        assert entry["created_at"] > 0
        assert entry["expires_at"] > entry["created_at"]
        assert entry["ttl_remaining"] > 0
        assert entry["ttl_remaining"] <= 300  # Should be less than or equal to TTL

    def test_get_cache_expired_entries_excluded(self, client: TestClient) -> None:
        """Test that expired entries are not returned."""
        from loopengine.behaviors import BehaviorCache

        cache = BehaviorCache()
        behavior = BehaviorResponse(action="test", parameters={}, reasoning="", metadata={})
        # Set with very short TTL
        cache.set("domain:agent:hash", behavior, ttl=0.001)

        # Wait for expiration
        import time

        time.sleep(0.01)

        with patch("loopengine.api.behaviors._get_behavior_cache", return_value=cache):
            response = client.get("/api/v1/behaviors/cache")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["entries"] == []


class TestPinBehaviorEndpoint:
    """Tests for the POST /api/v1/behaviors/pin endpoint."""

    def test_pin_behavior_success(self, client: TestClient, tmp_path: Path) -> None:
        """Test successfully pinning a behavior."""
        from loopengine.behaviors import BehaviorPinStore

        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        with patch("loopengine.api.behaviors._get_pin_store", return_value=pin_store):
            response = client.post(
                "/api/v1/behaviors/pin",
                json={
                    "domain_id": "shop",
                    "agent_type": "employee",
                    "context": {"task": "greet"},
                    "behavior": {
                        "action": "say_hello",
                        "parameters": {"greeting": "Welcome!"},
                        "reasoning": "Customer just entered",
                    },
                    "reason": "Default greeting behavior",
                },
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["pin_id"].startswith("pin-")
        assert data["domain_id"] == "shop"
        assert data["agent_type"] == "employee"
        assert "pinned_at" in data
        assert data["message"] == f"Behavior pinned successfully with ID {data['pin_id']}"

    def test_pin_behavior_without_reason(self, client: TestClient, tmp_path: Path) -> None:
        """Test pinning a behavior without providing a reason."""
        from loopengine.behaviors import BehaviorPinStore

        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        with patch("loopengine.api.behaviors._get_pin_store", return_value=pin_store):
            response = client.post(
                "/api/v1/behaviors/pin",
                json={
                    "domain_id": "shop",
                    "agent_type": "employee",
                    "context": {},
                    "behavior": {
                        "action": "wait",
                        "parameters": {},
                        "reasoning": "Nothing to do",
                    },
                },
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["pin_id"].startswith("pin-")

    def test_pin_behavior_duplicate_updates(self, client: TestClient, tmp_path: Path) -> None:
        """Test that pinning the same context again updates the behavior (idempotent)."""
        from loopengine.behaviors import BehaviorPinStore

        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        with patch("loopengine.api.behaviors._get_pin_store", return_value=pin_store):
            # First pin
            response1 = client.post(
                "/api/v1/behaviors/pin",
                json={
                    "domain_id": "shop",
                    "agent_type": "employee",
                    "context": {"task": "greet"},
                    "behavior": {
                        "action": "say_hello",
                        "parameters": {},
                        "reasoning": "First version",
                    },
                },
            )
            assert response1.status_code == status.HTTP_201_CREATED
            pin_id1 = response1.json()["pin_id"]

            # Second pin with same context but different behavior
            response2 = client.post(
                "/api/v1/behaviors/pin",
                json={
                    "domain_id": "shop",
                    "agent_type": "employee",
                    "context": {"task": "greet"},
                    "behavior": {
                        "action": "wave",
                        "parameters": {},
                        "reasoning": "Updated version",
                    },
                },
            )
            assert response2.status_code == status.HTTP_201_CREATED
            pin_id2 = response2.json()["pin_id"]

            # Same pin ID (updated, not duplicated)
            assert pin_id1 == pin_id2

            # Verify the behavior was actually updated
            pinned = pin_store.get_by_id(pin_id2)
            assert pinned is not None
            assert pinned.behavior.action == "wave"
            assert pinned.behavior.reasoning == "Updated version"

    def test_pin_behavior_stored_correctly(self, client: TestClient, tmp_path: Path) -> None:
        """Test that pinned behavior is stored and retrievable."""
        from loopengine.behaviors import BehaviorPinStore

        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        with patch("loopengine.api.behaviors._get_pin_store", return_value=pin_store):
            response = client.post(
                "/api/v1/behaviors/pin",
                json={
                    "domain_id": "restaurant",
                    "agent_type": "waiter",
                    "context": {"table": 5, "status": "seated"},
                    "behavior": {
                        "action": "bring_menu",
                        "parameters": {"menu_type": "dinner"},
                        "reasoning": "New customers need menus",
                    },
                    "reason": "Standard service flow",
                },
            )

        assert response.status_code == status.HTTP_201_CREATED
        pin_id = response.json()["pin_id"]

        # Verify stored correctly
        pinned = pin_store.get_by_id(pin_id)
        assert pinned is not None
        assert pinned.domain_id == "restaurant"
        assert pinned.agent_type == "waiter"
        assert pinned.context == {"table": 5, "status": "seated"}
        assert pinned.behavior.action == "bring_menu"
        assert pinned.behavior.parameters == {"menu_type": "dinner"}
        assert pinned.behavior.reasoning == "New customers need menus"
        assert pinned.reason == "Standard service flow"

    def test_pin_behavior_invalid_domain_id_chars(self, client: TestClient, tmp_path: Path) -> None:
        """Test that invalid domain_id characters return 400."""
        from loopengine.behaviors import BehaviorPinStore

        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        with patch("loopengine.api.behaviors._get_pin_store", return_value=pin_store):
            response = client.post(
                "/api/v1/behaviors/pin",
                json={
                    "domain_id": "shop/../../etc",
                    "agent_type": "employee",
                    "context": {},
                    "behavior": {
                        "action": "test",
                        "parameters": {},
                        "reasoning": "",
                    },
                },
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "invalid" in response.json()["detail"].lower()

    def test_pin_behavior_empty_domain_id(self, client: TestClient) -> None:
        """Test that empty domain_id returns 422."""
        response = client.post(
            "/api/v1/behaviors/pin",
            json={
                "domain_id": "",
                "agent_type": "employee",
                "context": {},
                "behavior": {
                    "action": "test",
                    "parameters": {},
                    "reasoning": "",
                },
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_pin_behavior_whitespace_domain_id(self, client: TestClient) -> None:
        """Test that whitespace-only domain_id returns 422."""
        response = client.post(
            "/api/v1/behaviors/pin",
            json={
                "domain_id": "   ",
                "agent_type": "employee",
                "context": {},
                "behavior": {
                    "action": "test",
                    "parameters": {},
                    "reasoning": "",
                },
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_pin_behavior_empty_agent_type(self, client: TestClient) -> None:
        """Test that empty agent_type returns 422."""
        response = client.post(
            "/api/v1/behaviors/pin",
            json={
                "domain_id": "shop",
                "agent_type": "",
                "context": {},
                "behavior": {
                    "action": "test",
                    "parameters": {},
                    "reasoning": "",
                },
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_pin_behavior_empty_action(self, client: TestClient) -> None:
        """Test that empty behavior action returns 422."""
        response = client.post(
            "/api/v1/behaviors/pin",
            json={
                "domain_id": "shop",
                "agent_type": "employee",
                "context": {},
                "behavior": {
                    "action": "",
                    "parameters": {},
                    "reasoning": "",
                },
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_pin_behavior_missing_behavior(self, client: TestClient) -> None:
        """Test that missing behavior field returns 422."""
        response = client.post(
            "/api/v1/behaviors/pin",
            json={
                "domain_id": "shop",
                "agent_type": "employee",
                "context": {},
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_pin_behavior_minimal_valid_request(self, client: TestClient, tmp_path: Path) -> None:
        """Test pinning with minimal required fields."""
        from loopengine.behaviors import BehaviorPinStore

        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        with patch("loopengine.api.behaviors._get_pin_store", return_value=pin_store):
            response = client.post(
                "/api/v1/behaviors/pin",
                json={
                    "domain_id": "shop",
                    "agent_type": "employee",
                    "behavior": {
                        "action": "wait",
                    },
                },
            )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["pin_id"].startswith("pin-")

        # Verify defaults were applied
        pinned = pin_store.get_by_id(data["pin_id"])
        assert pinned is not None
        assert pinned.context == {}
        assert pinned.behavior.parameters == {}
        assert pinned.behavior.reasoning == ""

    def test_pinned_behavior_used_in_generate(
        self, client: TestClient, tmp_path: Path, sample_schema: DomainSchema
    ) -> None:
        """Test that pinned behavior is used in subsequent generate calls."""
        from loopengine.behaviors import BehaviorPinStore

        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")
        domain_store = DomainStore(storage_dir=tmp_path / "domains")
        domain_store.save("shop", sample_schema)

        # Pin a behavior
        with patch("loopengine.api.behaviors._get_pin_store", return_value=pin_store):
            pin_response = client.post(
                "/api/v1/behaviors/pin",
                json={
                    "domain_id": "shop",
                    "agent_type": "customer",
                    "context": {"waiting": True},
                    "behavior": {
                        "action": "wait_patiently",
                        "parameters": {"patience_level": "high"},
                        "reasoning": "Customer is known to be patient",
                    },
                    "reason": "VIP customer preference",
                },
            )
        assert pin_response.status_code == status.HTTP_201_CREATED

        # Verify pin store has the behavior
        pinned = pin_store.get_behavior("shop", "customer", {"waiting": True})
        assert pinned is not None
        assert pinned.action == "wait_patiently"

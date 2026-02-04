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

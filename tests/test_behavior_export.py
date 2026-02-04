"""Tests for behavior history storage and export functionality."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import SecretStr

from loopengine.api.behaviors import (
    router,
    set_history_store,
)
from loopengine.behaviors import (
    BehaviorHistoryStore,
    DomainStore,
    StoredBehavior,
)
from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.domain_parser import AgentTypeSchema, DomainSchema
from loopengine.behaviors.llm_client import BehaviorResponse


@pytest.fixture
def history_store() -> BehaviorHistoryStore:
    """Create a fresh behavior history store."""
    return BehaviorHistoryStore(max_size=1000)


@pytest.fixture
def sample_behaviors() -> list[StoredBehavior]:
    """Create sample stored behaviors for testing."""
    base_time = time.time()
    return [
        StoredBehavior(
            timestamp=base_time - 100,
            domain_id="sandwich_shop",
            domain_type="sandwich shop",
            agent_type="sandwich_maker",
            agent_role="Makes sandwiches",
            action="make_sandwich",
            parameters={"type": "turkey"},
            reasoning="Customer ordered turkey sandwich",
            context={"pending_orders": 2},
            latency_ms=150.0,
            provider="claude",
            cached=False,
            fallback=False,
        ),
        StoredBehavior(
            timestamp=base_time - 50,
            domain_id="sandwich_shop",
            domain_type="sandwich shop",
            agent_type="cashier",
            agent_role="Takes orders",
            action="take_order",
            parameters={"customer_id": 1},
            reasoning="Customer is waiting",
            context={"queue_length": 3},
            latency_ms=120.0,
            provider="claude",
            cached=True,
            fallback=False,
        ),
        StoredBehavior(
            timestamp=base_time,
            domain_id="flower_shop",
            domain_type="flower shop",
            agent_type="florist",
            agent_role="Arranges flowers",
            action="arrange_bouquet",
            parameters={"flowers": ["roses", "lilies"]},
            reasoning="Customer wants a bouquet",
            context=None,
            latency_ms=200.0,
            provider="claude",
            cached=False,
            fallback=True,
        ),
    ]


class TestStoredBehavior:
    """Tests for the StoredBehavior dataclass."""

    def test_to_dict(self, sample_behaviors: list[StoredBehavior]) -> None:
        """Test converting to dictionary."""
        behavior = sample_behaviors[0]
        d = behavior.to_dict()

        assert d["domain_id"] == "sandwich_shop"
        assert d["agent_type"] == "sandwich_maker"
        assert d["action"] == "make_sandwich"
        assert d["parameters"] == {"type": "turkey"}
        assert d["reasoning"] == "Customer ordered turkey sandwich"
        assert d["latency_ms"] == 150.0
        assert d["cached"] is False

    def test_to_human_readable(self, sample_behaviors: list[StoredBehavior]) -> None:
        """Test converting to human-readable format."""
        behavior = sample_behaviors[0]
        text = behavior.to_human_readable()

        assert "sandwich_maker" in text
        assert "make_sandwich" in text
        assert "turkey" in text
        assert "150.00ms" in text
        assert "claude" in text.lower()


class TestBehaviorHistoryStore:
    """Tests for the BehaviorHistoryStore class."""

    def test_record_and_export(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test recording and exporting behaviors."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        exported = history_store.export()
        assert len(exported) == 3

        # Most recent first
        assert exported[0].domain_id == "flower_shop"
        assert exported[1].agent_type == "cashier"
        assert exported[2].agent_type == "sandwich_maker"

    def test_filter_by_domain_id(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test filtering by domain ID."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        exported = history_store.export(domain_id="sandwich_shop")
        assert len(exported) == 2
        assert all(b.domain_id == "sandwich_shop" for b in exported)

    def test_filter_by_domain_type(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test filtering by domain type."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        exported = history_store.export(domain_type="flower shop")
        assert len(exported) == 1
        assert exported[0].domain_type == "flower shop"

    def test_filter_by_agent_type(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test filtering by agent type."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        exported = history_store.export(agent_type="sandwich_maker")
        assert len(exported) == 1
        assert exported[0].agent_type == "sandwich_maker"

    def test_filter_by_time_range(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test filtering by time range."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        # Get the middle behavior's timestamp
        base_time = sample_behaviors[2].timestamp
        start_time = base_time - 75  # After first, includes second and third

        exported = history_store.export(start_time=start_time)
        assert len(exported) == 2
        assert all(b.timestamp >= start_time for b in exported)

    def test_pagination_limit(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test pagination with limit."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        exported = history_store.export(limit=2)
        assert len(exported) == 2
        assert exported[0].domain_id == "flower_shop"  # Most recent first

    def test_pagination_offset(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test pagination with offset."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        exported = history_store.export(offset=1, limit=2)
        assert len(exported) == 2
        assert exported[0].agent_type == "cashier"
        assert exported[1].agent_type == "sandwich_maker"

    def test_export_json(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test exporting as JSON."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        json_data = history_store.export_json()
        assert len(json_data) == 3
        assert isinstance(json_data[0], dict)
        assert json_data[0]["domain_id"] == "flower_shop"

    def test_export_human_readable(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test exporting as human-readable text."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        text = history_store.export_human_readable()
        assert "flower_shop" in text
        assert "sandwich_shop" in text
        assert "arrange_bouquet" in text
        assert "make_sandwich" in text
        assert "=" * 60 in text  # Separator

    def test_export_human_readable_empty(
        self,
        history_store: BehaviorHistoryStore,
    ) -> None:
        """Test human-readable export with no behaviors."""
        text = history_store.export_human_readable()
        assert "No behaviors found" in text

    def test_count(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test counting behaviors."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        assert history_store.count() == 3
        assert history_store.count(domain_id="sandwich_shop") == 2
        assert history_store.count(agent_type="florist") == 1

    def test_get_domains(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test getting unique domains."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        domains = history_store.get_domains()
        assert len(domains) == 2
        assert "sandwich_shop" in domains
        assert "flower_shop" in domains

    def test_get_agent_types(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test getting unique agent types."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        # All agent types
        agent_types = history_store.get_agent_types()
        assert len(agent_types) == 3
        assert "sandwich_maker" in agent_types
        assert "cashier" in agent_types
        assert "florist" in agent_types

        # Filter by domain
        agent_types = history_store.get_agent_types(domain_id="sandwich_shop")
        assert len(agent_types) == 2
        assert "florist" not in agent_types

    def test_clear(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test clearing history."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        assert history_store.size == 3
        history_store.clear()
        assert history_store.size == 0
        assert history_store.export() == []

    def test_max_size_limit(self) -> None:
        """Test that max size limit is enforced."""
        store = BehaviorHistoryStore(max_size=3)

        for i in range(5):
            store.record(
                StoredBehavior(
                    timestamp=time.time() + i,
                    domain_id=f"domain_{i}",
                    domain_type="test",
                    agent_type="agent",
                    agent_role="role",
                    action="action",
                )
            )

        assert store.size == 3
        assert store.total_recorded == 5
        domains = store.get_domains()
        assert "domain_0" not in domains
        assert "domain_1" not in domains
        assert "domain_4" in domains

    def test_get_stats(
        self,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test getting statistics."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        stats = history_store.get_stats()
        assert stats["size"] == 3
        assert stats["total_recorded"] == 3
        assert stats["max_size"] == 1000
        assert stats["unique_domains"] == 2
        assert stats["unique_agent_types"] == 3


# API Tests


@pytest.fixture
def sample_schema() -> DomainSchema:
    """Create a sample domain schema."""
    return DomainSchema(
        domain_type="sandwich shop",
        description="A sandwich shop",
        agent_types=[
            AgentTypeSchema(
                name="sandwich_maker",
                role="Makes sandwiches",
                capabilities=["make_sandwich"],
            ),
        ],
    )


@pytest.fixture
def mock_config() -> LLMConfig:
    """Create a mock LLM config."""
    return LLMConfig(
        llm_provider=LLMProvider.CLAUDE,
        anthropic_api_key=SecretStr("test-key"),
    )


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create temporary storage directory."""
    storage = tmp_path / "domains"
    storage.mkdir()
    return storage


@pytest.fixture
def app():
    """Create a FastAPI test app."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app, history_store: BehaviorHistoryStore) -> TestClient:
    """Create a test client with fresh history store."""
    set_history_store(history_store)
    return TestClient(app)


class TestExportEndpoint:
    """Tests for the GET /api/v1/behaviors/export endpoint."""

    def test_export_empty(self, client: TestClient) -> None:
        """Test exporting when no behaviors recorded."""
        response = client.get("/api/v1/behaviors/export")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["behaviors"] == []
        assert data["total_count"] == 0
        assert data["offset"] == 0
        assert data["limit"] == 100

    def test_export_with_behaviors(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test exporting recorded behaviors."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        response = client.get("/api/v1/behaviors/export")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["behaviors"]) == 3
        assert data["total_count"] == 3
        assert data["behaviors"][0]["domain_id"] == "flower_shop"

    def test_export_filter_domain_id(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test filtering by domain_id."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        response = client.get(
            "/api/v1/behaviors/export",
            params={"domain_id": "sandwich_shop"},
        )
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["behaviors"]) == 2
        assert data["total_count"] == 2
        assert all(b["domain_id"] == "sandwich_shop" for b in data["behaviors"])

    def test_export_filter_agent_type(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test filtering by agent_type."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        response = client.get(
            "/api/v1/behaviors/export",
            params={"agent_type": "florist"},
        )
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["behaviors"]) == 1
        assert data["behaviors"][0]["agent_type"] == "florist"

    def test_export_filter_time_range(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test filtering by time range."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        base_time = sample_behaviors[2].timestamp
        response = client.get(
            "/api/v1/behaviors/export",
            params={
                "start_time": base_time - 75,
                "end_time": base_time + 1,
            },
        )
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["behaviors"]) == 2

    def test_export_pagination(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test pagination parameters."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        response = client.get(
            "/api/v1/behaviors/export",
            params={"limit": 2, "offset": 1},
        )
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert len(data["behaviors"]) == 2
        assert data["total_count"] == 3
        assert data["offset"] == 1
        assert data["limit"] == 2


class TestExportTextEndpoint:
    """Tests for the GET /api/v1/behaviors/export/text endpoint."""

    def test_export_text_empty(self, client: TestClient) -> None:
        """Test text export when no behaviors recorded."""
        response = client.get("/api/v1/behaviors/export/text")
        assert response.status_code == status.HTTP_200_OK
        assert "No behaviors found" in response.text

    def test_export_text_with_behaviors(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test text export with behaviors."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        response = client.get("/api/v1/behaviors/export/text")
        assert response.status_code == status.HTTP_200_OK
        assert "text/plain" in response.headers["content-type"]
        assert "flower_shop" in response.text
        assert "sandwich_shop" in response.text
        assert "arrange_bouquet" in response.text


class TestExportStatsEndpoint:
    """Tests for the GET /api/v1/behaviors/export/stats endpoint."""

    def test_export_stats_empty(self, client: TestClient) -> None:
        """Test stats when empty."""
        response = client.get("/api/v1/behaviors/export/stats")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["size"] == 0
        assert data["total_recorded"] == 0

    def test_export_stats_with_behaviors(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test stats with behaviors."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        response = client.get("/api/v1/behaviors/export/stats")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["size"] == 3
        assert data["total_recorded"] == 3
        assert data["unique_domains"] == 2
        assert data["unique_agent_types"] == 3


class TestExportDomainsEndpoint:
    """Tests for the GET /api/v1/behaviors/export/domains endpoint."""

    def test_export_domains_empty(self, client: TestClient) -> None:
        """Test domains list when empty."""
        response = client.get("/api/v1/behaviors/export/domains")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_export_domains_with_behaviors(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test domains list with behaviors."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        response = client.get("/api/v1/behaviors/export/domains")
        assert response.status_code == status.HTTP_200_OK

        domains = response.json()
        assert len(domains) == 2
        assert "sandwich_shop" in domains
        assert "flower_shop" in domains


class TestExportAgentTypesEndpoint:
    """Tests for the GET /api/v1/behaviors/export/agent-types endpoint."""

    def test_export_agent_types_empty(self, client: TestClient) -> None:
        """Test agent types list when empty."""
        response = client.get("/api/v1/behaviors/export/agent-types")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_export_agent_types_with_behaviors(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test agent types list with behaviors."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        response = client.get("/api/v1/behaviors/export/agent-types")
        assert response.status_code == status.HTTP_200_OK

        agent_types = response.json()
        assert len(agent_types) == 3
        assert "sandwich_maker" in agent_types
        assert "cashier" in agent_types
        assert "florist" in agent_types

    def test_export_agent_types_filtered(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test agent types list filtered by domain."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        response = client.get(
            "/api/v1/behaviors/export/agent-types",
            params={"domain_id": "sandwich_shop"},
        )
        assert response.status_code == status.HTTP_200_OK

        agent_types = response.json()
        assert len(agent_types) == 2
        assert "florist" not in agent_types


class TestClearHistoryEndpoint:
    """Tests for the DELETE /api/v1/behaviors/export endpoint."""

    def test_clear_history(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        sample_behaviors: list[StoredBehavior],
    ) -> None:
        """Test clearing behavior history."""
        for behavior in sample_behaviors:
            history_store.record(behavior)

        assert history_store.size == 3

        response = client.delete("/api/v1/behaviors/export")
        assert response.status_code == status.HTTP_200_OK
        assert "cleared" in response.json()["message"].lower()

        assert history_store.size == 0


class TestBehaviorRecordingIntegration:
    """Tests for behavior recording during generation."""

    def test_behavior_recorded_on_generate(
        self,
        client: TestClient,
        history_store: BehaviorHistoryStore,
        temp_storage_dir: Path,
        sample_schema: DomainSchema,
    ) -> None:
        """Test that behaviors are recorded when generated."""
        store = DomainStore(storage_dir=temp_storage_dir)
        store.save("sandwich_shop", sample_schema)

        mock_response = BehaviorResponse(
            action="make_sandwich",
            parameters={"type": "turkey"},
            reasoning="Customer ordered turkey",
            metadata={"latency_ms": 150.0, "provider": "claude", "cached": False},
        )

        mock_engine = MagicMock()
        mock_engine.generate_behavior.return_value = mock_response
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

        # Check behavior was recorded
        assert history_store.size == 1
        exported = history_store.export()
        assert len(exported) == 1
        assert exported[0].domain_id == "sandwich_shop"
        assert exported[0].action == "make_sandwich"
        assert exported[0].reasoning == "Customer ordered turkey"
        assert exported[0].context == {"pending_orders": 2}

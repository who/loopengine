"""Tests for the metrics API endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from loopengine.api import metrics as metrics_module
from loopengine.behaviors import AIBehaviorEngine, LatencyTracker
from loopengine.server.app import app


@pytest.fixture
def client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock AIBehaviorEngine."""
    engine = MagicMock(spec=AIBehaviorEngine)
    tracker = LatencyTracker()
    engine.latency_tracker = tracker
    engine.metrics = {
        "total_queries": 100,
        "total_latency_ms": 15000.0,
        "avg_latency_ms": 150.0,
        "provider": "claude",
        "rate_limit_events": 5,
        "concurrent_requests": 3,
        "peak_concurrent_requests": 10,
        "max_concurrent_limit": 50,
        "rate_limit_stats": {
            "total_retries": 10,
            "fallbacks_used": 2,
        },
        "p50_latency_ms": 100.0,
        "p95_latency_ms": 500.0,
        "p99_latency_ms": 800.0,
        "min_latency_ms": 50.0,
        "max_latency_ms": 1200.0,
        "slow_query_count": 5,
        "slow_query_threshold_ms": 3000.0,
    }
    return engine


@pytest.fixture
def setup_mock_engine(mock_engine: MagicMock) -> MagicMock:
    """Set up mock engine and clean up after test."""
    original = metrics_module._engine
    metrics_module.set_engine(mock_engine)
    yield mock_engine
    metrics_module._engine = original


class TestGetMetrics:
    """Tests for GET /api/v1/metrics endpoint."""

    def test_get_metrics_success(self, client: TestClient, setup_mock_engine: MagicMock) -> None:
        """Test successful metrics retrieval."""
        # Add some latencies to the tracker
        tracker = setup_mock_engine.latency_tracker
        for i in range(100):
            tracker.record(float(i * 10))  # 0, 10, 20, ..., 990

        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "provider" in data
        assert "latency" in data
        assert "concurrency" in data
        assert "rate_limits" in data

    def test_get_metrics_latency_fields(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test metrics includes all latency fields."""
        tracker = setup_mock_engine.latency_tracker
        for i in range(100):
            tracker.record(float(i * 10))

        response = client.get("/api/v1/metrics")
        data = response.json()

        latency = data["latency"]
        assert "total_queries" in latency
        assert "avg_latency_ms" in latency
        assert "p50_latency_ms" in latency
        assert "p95_latency_ms" in latency
        assert "p99_latency_ms" in latency
        assert "min_latency_ms" in latency
        assert "max_latency_ms" in latency
        assert "slow_query_count" in latency
        assert "slow_query_threshold_ms" in latency
        assert "history_size" in latency

    def test_get_metrics_concurrency_fields(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test metrics includes all concurrency fields."""
        response = client.get("/api/v1/metrics")
        data = response.json()

        concurrency = data["concurrency"]
        assert "concurrent_requests" in concurrency
        assert "peak_concurrent_requests" in concurrency
        assert "max_concurrent_limit" in concurrency

    def test_get_metrics_rate_limit_fields(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test metrics includes all rate limit fields."""
        response = client.get("/api/v1/metrics")
        data = response.json()

        rate_limits = data["rate_limits"]
        assert "rate_limit_events" in rate_limits
        assert "total_retries" in rate_limits
        assert "fallbacks_used" in rate_limits

    def test_get_metrics_empty_tracker(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test metrics with empty tracker returns zeros."""
        response = client.get("/api/v1/metrics")
        data = response.json()

        latency = data["latency"]
        assert latency["total_queries"] == 0
        assert latency["avg_latency_ms"] == 0.0
        assert latency["p50_latency_ms"] == 0.0
        assert latency["history_size"] == 0


class TestGetSlowQueries:
    """Tests for GET /api/v1/metrics/slow-queries endpoint."""

    def test_get_slow_queries_empty(self, client: TestClient, setup_mock_engine: MagicMock) -> None:
        """Test getting slow queries when none exist."""
        response = client.get("/api/v1/metrics/slow-queries")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_slow_queries_with_data(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test getting slow queries with data."""
        tracker = setup_mock_engine.latency_tracker
        tracker._slow_threshold_ms = 1000.0  # Low threshold for testing

        # Add some slow queries
        tracker.record(1500.0, "cashier", "sandwich_shop", {"order_id": 1})
        tracker.record(2000.0, "sandwich_maker", "sandwich_shop", {"order_id": 2})

        response = client.get("/api/v1/metrics/slow-queries")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2

        # Check structure
        query = data[0]
        assert "timestamp" in query
        assert "latency_ms" in query
        assert "agent_type" in query
        assert "domain_type" in query
        assert "context" in query

    def test_get_slow_queries_with_limit(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test getting slow queries with limit parameter."""
        tracker = setup_mock_engine.latency_tracker
        tracker._slow_threshold_ms = 1000.0

        # Add many slow queries
        for i in range(20):
            tracker.record(1500.0 + i, f"agent_{i}", "shop")

        response = client.get("/api/v1/metrics/slow-queries?limit=5")
        assert response.status_code == 200
        assert len(response.json()) == 5

    def test_get_slow_queries_context_included(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test slow queries include context for debugging."""
        tracker = setup_mock_engine.latency_tracker
        tracker._slow_threshold_ms = 1000.0

        context = {"agent_id": "agent_123", "action": "process_order"}
        tracker.record(1500.0, "cashier", "shop", context)

        response = client.get("/api/v1/metrics/slow-queries")
        data = response.json()

        assert len(data) == 1
        assert data[0]["context"]["agent_id"] == "agent_123"
        assert data[0]["context"]["action"] == "process_order"


class TestGetAlerts:
    """Tests for GET /api/v1/metrics/alerts endpoint."""

    def test_get_alerts_empty(self, client: TestClient, setup_mock_engine: MagicMock) -> None:
        """Test getting alerts when none exist."""
        response = client.get("/api/v1/metrics/alerts")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_alerts_with_data(self, client: TestClient, setup_mock_engine: MagicMock) -> None:
        """Test getting alerts with data."""
        tracker = setup_mock_engine.latency_tracker
        tracker._slow_threshold_ms = 100.0
        tracker._alert_window_size = 10
        tracker._alert_threshold_pct = 10.0
        tracker._alert_cooldown_seconds = 0

        # Trigger an alert
        for _ in range(8):
            tracker.record(50.0)
        for _ in range(3):
            tracker.record(150.0)

        response = client.get("/api/v1/metrics/alerts")
        assert response.status_code == 200

        data = response.json()
        assert len(data) >= 1

        # Check structure
        alert = data[0]
        assert "timestamp" in alert
        assert "severity" in alert
        assert "message" in alert
        assert "p95_latency_ms" in alert
        assert "window_size" in alert
        assert "slow_query_count" in alert

    def test_get_alerts_with_limit(self, client: TestClient, setup_mock_engine: MagicMock) -> None:
        """Test getting alerts with limit parameter."""
        tracker = setup_mock_engine.latency_tracker
        tracker._slow_threshold_ms = 100.0
        tracker._alert_window_size = 10
        tracker._alert_threshold_pct = 10.0
        tracker._alert_cooldown_seconds = 0

        # Trigger multiple alerts
        for _ in range(10):
            for _ in range(8):
                tracker.record(50.0)
            for _ in range(3):
                tracker.record(150.0)
            tracker._last_alert_time = 0

        response = client.get("/api/v1/metrics/alerts?limit=3")
        assert response.status_code == 200
        assert len(response.json()) <= 3

    def test_get_alerts_severity_values(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test alert severity values are valid."""
        tracker = setup_mock_engine.latency_tracker
        tracker._slow_threshold_ms = 100.0
        tracker._alert_window_size = 10
        tracker._alert_threshold_pct = 10.0
        tracker._alert_cooldown_seconds = 0

        # Trigger an alert
        for _ in range(8):
            tracker.record(50.0)
        for _ in range(5):  # High percentage for critical
            tracker.record(150.0)

        response = client.get("/api/v1/metrics/alerts")
        data = response.json()

        for alert in data:
            assert alert["severity"] in ["warning", "critical"]


class TestResetMetrics:
    """Tests for POST /api/v1/metrics/reset endpoint."""

    def test_reset_metrics(self, client: TestClient, setup_mock_engine: MagicMock) -> None:
        """Test resetting metrics."""
        tracker = setup_mock_engine.latency_tracker
        tracker._slow_threshold_ms = 100.0

        # Add some data
        for _ in range(50):
            tracker.record(150.0)

        # Reset
        response = client.post("/api/v1/metrics/reset")
        assert response.status_code == 200
        assert "message" in response.json()

        # Verify engine.reset_metrics was called
        setup_mock_engine.reset_metrics.assert_called_once()


class TestMetricsIntegration:
    """Integration tests for metrics endpoint."""

    def test_metrics_after_multiple_requests(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test metrics accuracy after multiple requests (acceptance criterion)."""
        tracker = setup_mock_engine.latency_tracker

        # Simulate 100 requests
        for i in range(100):
            latency = 100.0 + (i % 10) * 50  # Varying latencies
            tracker.record(latency)

        response = client.get("/api/v1/metrics")
        data = response.json()

        assert data["latency"]["total_queries"] == 100
        assert data["latency"]["history_size"] == 100

    def test_slow_queries_over_3_seconds_tracked(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test slow queries over 3 seconds are tracked (acceptance criterion)."""
        tracker = setup_mock_engine.latency_tracker
        # Default threshold is 3000ms

        # Normal query
        tracker.record(500.0, "fast_agent", "shop")

        # Slow queries
        tracker.record(3100.0, "slow_agent_1", "shop", {"order": 1})
        tracker.record(4000.0, "slow_agent_2", "shop", {"order": 2})
        tracker.record(5000.0, "slow_agent_3", "shop", {"order": 3})

        response = client.get("/api/v1/metrics/slow-queries")
        data = response.json()

        assert len(data) == 3
        # All slow queries have latency > 3000ms
        for query in data:
            assert query["latency_ms"] > 3000.0

    def test_metrics_returns_percentiles(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test GET /metrics returns latency percentiles (acceptance criterion)."""
        tracker = setup_mock_engine.latency_tracker

        # Add diverse latencies
        for i in range(100):
            tracker.record(float(i * 10))

        response = client.get("/api/v1/metrics")
        data = response.json()

        latency = data["latency"]
        # Percentiles should be present and reasonable
        assert latency["p50_latency_ms"] > 0
        assert latency["p95_latency_ms"] > latency["p50_latency_ms"]
        assert latency["p99_latency_ms"] >= latency["p95_latency_ms"]

    def test_alert_mechanism_for_sustained_high_latency(
        self, client: TestClient, setup_mock_engine: MagicMock
    ) -> None:
        """Test alert triggers on sustained high latency (acceptance criterion)."""
        tracker = setup_mock_engine.latency_tracker
        tracker._slow_threshold_ms = 3000.0
        tracker._alert_window_size = 20
        tracker._alert_threshold_pct = 10.0
        tracker._alert_cooldown_seconds = 0

        # Simulate sustained high latency (20% slow)
        for _ in range(16):  # 80% normal
            tracker.record(500.0)
        for _ in range(4):  # 20% slow
            tracker.record(3500.0)

        response = client.get("/api/v1/metrics/alerts")
        data = response.json()

        assert len(data) >= 1
        alert = data[0]
        assert alert["slow_query_count"] >= 4

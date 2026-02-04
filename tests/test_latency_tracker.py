"""Tests for the latency tracker module."""

import threading
from unittest.mock import MagicMock

from loopengine.behaviors.latency_tracker import (
    SLOW_QUERY_THRESHOLD_MS,
    AlertSeverity,
    LatencyAlert,
    LatencyTracker,
    SlowQueryEvent,
)


class TestSlowQueryEvent:
    """Tests for SlowQueryEvent dataclass."""

    def test_create_event(self) -> None:
        """Test creating a slow query event."""
        event = SlowQueryEvent(
            timestamp=1234567890.0,
            latency_ms=3500.0,
            agent_type="cashier",
            domain_type="sandwich_shop",
            context={"order_id": 123},
        )
        assert event.timestamp == 1234567890.0
        assert event.latency_ms == 3500.0
        assert event.agent_type == "cashier"
        assert event.domain_type == "sandwich_shop"
        assert event.context == {"order_id": 123}

    def test_event_default_context(self) -> None:
        """Test event with default empty context."""
        event = SlowQueryEvent(
            timestamp=1234567890.0,
            latency_ms=3500.0,
            agent_type="cashier",
            domain_type="sandwich_shop",
        )
        assert event.context == {}


class TestLatencyAlert:
    """Tests for LatencyAlert dataclass."""

    def test_create_alert(self) -> None:
        """Test creating a latency alert."""
        alert = LatencyAlert(
            timestamp=1234567890.0,
            severity=AlertSeverity.WARNING,
            message="High latency detected",
            p95_latency_ms=3200.0,
            window_size=100,
            slow_query_count=15,
        )
        assert alert.timestamp == 1234567890.0
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "High latency detected"
        assert alert.p95_latency_ms == 3200.0
        assert alert.window_size == 100
        assert alert.slow_query_count == 15


class TestLatencyTracker:
    """Tests for LatencyTracker class."""

    def test_default_initialization(self) -> None:
        """Test default tracker initialization."""
        tracker = LatencyTracker()
        assert tracker.slow_threshold_ms == SLOW_QUERY_THRESHOLD_MS
        stats = tracker.get_stats()
        assert stats["total_queries"] == 0
        assert stats["history_size"] == 0

    def test_custom_initialization(self) -> None:
        """Test custom tracker initialization."""
        callback = MagicMock()
        tracker = LatencyTracker(
            max_history=500,
            slow_threshold_ms=2000.0,
            alert_callback=callback,
            alert_window_size=50,
            alert_threshold_pct=5.0,
        )
        assert tracker.slow_threshold_ms == 2000.0

    def test_record_latency(self) -> None:
        """Test recording a single latency."""
        tracker = LatencyTracker()
        tracker.record(100.0, "cashier", "sandwich_shop")
        stats = tracker.get_stats()
        assert stats["total_queries"] == 1
        assert stats["history_size"] == 1
        assert stats["avg_latency_ms"] == 100.0

    def test_record_multiple_latencies(self) -> None:
        """Test recording multiple latencies."""
        tracker = LatencyTracker()
        tracker.record(100.0)
        tracker.record(200.0)
        tracker.record(300.0)
        stats = tracker.get_stats()
        assert stats["total_queries"] == 3
        assert stats["avg_latency_ms"] == 200.0

    def test_percentile_calculation(self) -> None:
        """Test p50, p95, p99 percentile calculations."""
        tracker = LatencyTracker(max_history=1000)
        # Add 100 latencies: 1, 2, 3, ..., 100
        for i in range(1, 101):
            tracker.record(float(i))

        stats = tracker.get_stats()
        # Percentiles use integer index, so allow slight variance
        assert 49.0 <= stats["p50_latency_ms"] <= 51.0
        assert 94.0 <= stats["p95_latency_ms"] <= 96.0
        assert 98.0 <= stats["p99_latency_ms"] <= 100.0
        assert stats["min_latency_ms"] == 1.0
        assert stats["max_latency_ms"] == 100.0

    def test_slow_query_detection(self) -> None:
        """Test slow query detection (>3 seconds)."""
        tracker = LatencyTracker(slow_threshold_ms=3000.0)
        # Normal query
        tracker.record(100.0, "cashier", "sandwich_shop")
        assert len(tracker.get_slow_queries()) == 0

        # Slow query
        tracker.record(3500.0, "sandwich_maker", "sandwich_shop", {"order_id": 1})
        slow_queries = tracker.get_slow_queries()
        assert len(slow_queries) == 1
        assert slow_queries[0].latency_ms == 3500.0
        assert slow_queries[0].agent_type == "sandwich_maker"

    def test_slow_query_context(self) -> None:
        """Test slow query includes context for debugging."""
        tracker = LatencyTracker(slow_threshold_ms=1000.0)
        context = {"order_id": 123, "customer": "test"}
        tracker.record(1500.0, "cashier", "shop", context)

        slow_queries = tracker.get_slow_queries()
        assert len(slow_queries) == 1
        assert slow_queries[0].context == context

    def test_slow_query_ordering(self) -> None:
        """Test slow queries returned most recent first."""
        tracker = LatencyTracker(slow_threshold_ms=1000.0)
        tracker.record(1100.0, "first", "shop")
        tracker.record(1200.0, "second", "shop")
        tracker.record(1300.0, "third", "shop")

        slow_queries = tracker.get_slow_queries()
        assert len(slow_queries) == 3
        assert slow_queries[0].agent_type == "third"  # Most recent
        assert slow_queries[2].agent_type == "first"  # Oldest

    def test_slow_query_limit(self) -> None:
        """Test slow query limit parameter."""
        tracker = LatencyTracker(slow_threshold_ms=1000.0)
        for i in range(10):
            tracker.record(1500.0 + i, f"agent_{i}", "shop")

        limited = tracker.get_slow_queries(limit=5)
        assert len(limited) == 5

    def test_alert_triggered_on_sustained_high_latency(self) -> None:
        """Test alert is triggered when sustained high latency detected."""
        callback = MagicMock()
        tracker = LatencyTracker(
            slow_threshold_ms=100.0,  # Low threshold for testing
            alert_callback=callback,
            alert_window_size=10,
            alert_threshold_pct=10.0,  # 10% threshold
        )
        tracker._alert_cooldown_seconds = 0  # Disable cooldown for testing

        # Add 10 normal latencies
        for _ in range(10):
            tracker.record(50.0)

        # No alert yet
        assert callback.call_count == 0

        # Add 2 slow queries (20% of last 10)
        for _ in range(8):
            tracker.record(50.0)
        for _ in range(2):
            tracker.record(150.0)

        # Should trigger at least one alert (may trigger on each slow query)
        assert callback.call_count >= 1
        alert = callback.call_args[0][0]
        assert isinstance(alert, LatencyAlert)
        assert alert.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]

    def test_alert_severity_levels(self) -> None:
        """Test alert severity based on slow query percentage."""
        tracker = LatencyTracker(
            slow_threshold_ms=100.0,
            alert_window_size=10,
            alert_threshold_pct=10.0,
        )
        tracker._alert_cooldown_seconds = 0

        # 20%+ should be critical
        for _ in range(8):
            tracker.record(50.0)
        for _ in range(3):  # More than 20%
            tracker.record(150.0)

        alerts = tracker.get_alerts()
        # Find the critical alert
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) >= 1

    def test_alert_cooldown(self) -> None:
        """Test alert cooldown prevents spam."""
        callback = MagicMock()
        tracker = LatencyTracker(
            slow_threshold_ms=100.0,
            alert_callback=callback,
            alert_window_size=10,
            alert_threshold_pct=10.0,
        )
        tracker._alert_cooldown_seconds = 60  # 60 second cooldown

        # Trigger first alert
        for _ in range(8):
            tracker.record(50.0)
        for _ in range(3):
            tracker.record(150.0)

        # Continue with slow queries - should not trigger due to cooldown
        for _ in range(10):
            tracker.record(150.0)

        # Only one alert despite continued high latency
        assert callback.call_count == 1

    def test_get_alerts_ordering(self) -> None:
        """Test alerts returned most recent first."""
        tracker = LatencyTracker(
            slow_threshold_ms=100.0,
            alert_window_size=10,
            alert_threshold_pct=10.0,
        )
        tracker._alert_cooldown_seconds = 0

        # Trigger multiple alerts
        for _ in range(3):
            for _ in range(8):
                tracker.record(50.0)
            for _ in range(3):
                tracker.record(150.0)
            tracker._last_alert_time = 0  # Reset cooldown

        alerts = tracker.get_alerts()
        # Most recent alert should be first
        if len(alerts) > 1:
            assert alerts[0].timestamp >= alerts[1].timestamp

    def test_get_alerts_limit(self) -> None:
        """Test alerts limit parameter."""
        tracker = LatencyTracker(
            slow_threshold_ms=100.0,
            alert_window_size=10,
            alert_threshold_pct=10.0,
        )
        tracker._alert_cooldown_seconds = 0

        # Trigger many alerts
        for _ in range(10):
            for _ in range(8):
                tracker.record(50.0)
            for _ in range(3):
                tracker.record(150.0)
            tracker._last_alert_time = 0

        limited = tracker.get_alerts(limit=5)
        assert len(limited) <= 5

    def test_stats_empty_tracker(self) -> None:
        """Test stats on empty tracker returns zeros."""
        tracker = LatencyTracker()
        stats = tracker.get_stats()
        assert stats["total_queries"] == 0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["p50_latency_ms"] == 0.0
        assert stats["p95_latency_ms"] == 0.0
        assert stats["p99_latency_ms"] == 0.0
        assert stats["min_latency_ms"] == 0.0
        assert stats["max_latency_ms"] == 0.0
        assert stats["history_size"] == 0

    def test_stats_includes_threshold(self) -> None:
        """Test stats includes slow query threshold."""
        tracker = LatencyTracker(slow_threshold_ms=2500.0)
        stats = tracker.get_stats()
        assert stats["slow_query_threshold_ms"] == 2500.0

    def test_history_limit_enforced(self) -> None:
        """Test max_history limit is enforced."""
        tracker = LatencyTracker(max_history=10)
        for i in range(20):
            tracker.record(float(i))

        stats = tracker.get_stats()
        assert stats["history_size"] == 10
        # Total queries still counts all
        assert stats["total_queries"] == 20

    def test_reset_clears_all_data(self) -> None:
        """Test reset clears all tracked data."""
        tracker = LatencyTracker(slow_threshold_ms=100.0, alert_window_size=10)
        tracker._alert_cooldown_seconds = 0

        # Add data
        for _ in range(20):
            tracker.record(150.0)

        # Verify data exists
        assert tracker.get_stats()["total_queries"] > 0
        assert len(tracker.get_slow_queries()) > 0

        # Reset
        tracker.reset()

        # Verify cleared
        stats = tracker.get_stats()
        assert stats["total_queries"] == 0
        assert stats["history_size"] == 0
        assert len(tracker.get_slow_queries()) == 0
        assert len(tracker.get_alerts()) == 0

    def test_set_alert_callback(self) -> None:
        """Test setting alert callback."""
        tracker = LatencyTracker(
            slow_threshold_ms=100.0,
            alert_window_size=10,
            alert_threshold_pct=10.0,
        )
        tracker._alert_cooldown_seconds = 0

        callback = MagicMock()
        tracker.set_alert_callback(callback)

        # Trigger alert
        for _ in range(8):
            tracker.record(50.0)
        for _ in range(3):
            tracker.record(150.0)

        assert callback.call_count >= 1

    def test_set_slow_threshold(self) -> None:
        """Test setting slow threshold."""
        tracker = LatencyTracker(slow_threshold_ms=3000.0)
        assert tracker.slow_threshold_ms == 3000.0

        tracker.slow_threshold_ms = 5000.0
        assert tracker.slow_threshold_ms == 5000.0

    def test_thread_safety(self) -> None:
        """Test tracker is thread-safe under concurrent access."""
        tracker = LatencyTracker(max_history=1000)
        num_threads = 10
        records_per_thread = 100

        def record_latencies() -> None:
            for i in range(records_per_thread):
                tracker.record(float(i), f"agent_{threading.current_thread().name}", "shop")

        threads = [threading.Thread(target=record_latencies) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = tracker.get_stats()
        assert stats["total_queries"] == num_threads * records_per_thread

    def test_thread_safety_concurrent_stats(self) -> None:
        """Test getting stats while recording is thread-safe."""
        tracker = LatencyTracker(max_history=1000)
        errors: list[Exception] = []

        def record_latencies() -> None:
            try:
                for i in range(100):
                    tracker.record(float(i))
            except Exception as e:
                errors.append(e)

        def read_stats() -> None:
            try:
                for _ in range(100):
                    tracker.get_stats()
                    tracker.get_slow_queries()
                    tracker.get_alerts()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_latencies) for _ in range(5)] + [
            threading.Thread(target=read_stats) for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_metrics_accuracy_after_100_requests(self) -> None:
        """Test latency stats are accurate after 100 requests (acceptance criterion)."""
        tracker = LatencyTracker(max_history=1000)

        # Generate known distribution: 90 fast, 5 medium, 5 slow
        latencies = [100.0] * 90 + [1500.0] * 5 + [4000.0] * 5
        for lat in latencies:
            tracker.record(lat)

        stats = tracker.get_stats()
        assert stats["total_queries"] == 100
        # p95 should be around or above 1500 (5th percentile from top)
        assert stats["p95_latency_ms"] >= 1500.0
        # p50 should be 100 (median of mostly fast queries)
        assert stats["p50_latency_ms"] == 100.0

    def test_slow_queries_logged_with_context(self) -> None:
        """Test slow queries (>3 seconds) are logged with context (acceptance criterion)."""
        tracker = LatencyTracker(slow_threshold_ms=3000.0)

        # Record slow query with context
        context = {
            "agent_id": "agent_123",
            "domain_type": "sandwich_shop",
            "order_id": 456,
        }
        tracker.record(
            latency_ms=3500.0,
            agent_type="cashier",
            domain_type="sandwich_shop",
            context=context,
        )

        slow_queries = tracker.get_slow_queries()
        assert len(slow_queries) == 1
        query = slow_queries[0]
        assert query.latency_ms == 3500.0
        assert query.agent_type == "cashier"
        assert query.domain_type == "sandwich_shop"
        assert query.context["agent_id"] == "agent_123"
        assert query.context["order_id"] == 456

    def test_alert_on_simulated_slow_responses(self) -> None:
        """Test alert triggers on simulated slow responses (acceptance criterion)."""
        alerts_received: list[LatencyAlert] = []

        def alert_callback(alert: LatencyAlert) -> None:
            alerts_received.append(alert)

        tracker = LatencyTracker(
            slow_threshold_ms=3000.0,  # NFR-001: 3 second threshold
            alert_callback=alert_callback,
            alert_window_size=20,
            alert_threshold_pct=10.0,  # Alert if >10% slow
        )
        tracker._alert_cooldown_seconds = 0

        # Simulate mostly good responses with some slow ones
        for _ in range(16):  # 80% fast
            tracker.record(500.0)
        for _ in range(4):  # 20% slow
            tracker.record(3500.0)

        assert len(alerts_received) >= 1
        alert = alerts_received[0]
        assert "high latency" in alert.message.lower()
        assert alert.slow_query_count == 4

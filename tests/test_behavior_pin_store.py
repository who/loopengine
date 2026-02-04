"""Tests for the BehaviorPinStore module."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from loopengine.behaviors.behavior_pin_store import (
    BehaviorPinStore,
    BehaviorPinStoreError,
    PinnedBehavior,
)
from loopengine.behaviors.cache_key_generator import generate_key
from loopengine.behaviors.llm_client import BehaviorResponse


@pytest.fixture
def behavior_response() -> BehaviorResponse:
    """Create a test behavior response."""
    return BehaviorResponse(
        action="make_sandwich",
        parameters={"type": "turkey", "extras": ["lettuce", "tomato"]},
        reasoning="Customer ordered a turkey sandwich",
        metadata={"model": "test-model"},
    )


@pytest.fixture
def pin_store(tmp_path: Path) -> BehaviorPinStore:
    """Create a test pin store with temporary storage."""
    return BehaviorPinStore(storage_dir=tmp_path / "pins")


class TestBehaviorPinStoreInit:
    """Tests for BehaviorPinStore initialization."""

    def test_init_with_default_path(self) -> None:
        """Test store initializes with default path."""
        store = BehaviorPinStore()
        assert store._storage_dir == Path("data/pins")

    def test_init_with_custom_path(self, tmp_path: Path) -> None:
        """Test store initializes with custom path."""
        custom_path = tmp_path / "custom_pins"
        store = BehaviorPinStore(storage_dir=custom_path)
        assert store._storage_dir == custom_path

    def test_init_loads_existing_pins(
        self, tmp_path: Path, behavior_response: BehaviorResponse
    ) -> None:
        """Test store loads existing pins on init."""
        pins_dir = tmp_path / "pins"
        pins_dir.mkdir(parents=True)

        # Create a pin file manually
        pin_data = {
            "pins": [
                {
                    "pin_id": "pin-test123",
                    "domain_id": "shop",
                    "agent_type": "employee",
                    "context": {"task": "wait"},
                    "behavior": behavior_response.model_dump(),
                    "cache_key": generate_key("shop", "employee", {"task": "wait"}),
                    "pinned_at": "2024-01-01T00:00:00+00:00",
                    "reason": "test pin",
                }
            ]
        }
        with open(pins_dir / "shop_pins.json", "w") as f:
            json.dump(pin_data, f)

        store = BehaviorPinStore(storage_dir=pins_dir)

        assert store.total_pins == 1
        assert store.get_by_id("pin-test123") is not None

    def test_init_creates_empty_indices(self, pin_store: BehaviorPinStore) -> None:
        """Test fresh store has empty indices."""
        assert pin_store.total_pins == 0
        assert len(pin_store.list_pins()) == 0


class TestBehaviorPinStorePin:
    """Tests for pin operation."""

    def test_pin_creates_new_entry(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test pin creates a new pinned behavior."""
        pin_id = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
            reason="Good behavior",
        )

        assert pin_id.startswith("pin-")
        assert pin_store.total_pins == 1

    def test_pin_returns_existing_id_for_duplicate(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test pin returns existing ID for same domain/agent/context."""
        pin_id1 = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        new_response = BehaviorResponse(
            action="different_action",
            parameters={},
            reasoning="Different",
        )
        pin_id2 = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=new_response,
        )

        assert pin_id1 == pin_id2
        assert pin_store.total_pins == 1

        # Verify behavior was updated
        pinned = pin_store.get_by_id(pin_id1)
        assert pinned is not None
        assert pinned.behavior.action == "different_action"

    def test_pin_different_contexts_create_separate_pins(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test different contexts create separate pins."""
        pin_id1 = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )
        pin_id2 = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "serve"},
            behavior=behavior_response,
        )

        assert pin_id1 != pin_id2
        assert pin_store.total_pins == 2

    def test_pin_validates_domain_id(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test pin validates domain ID format."""
        with pytest.raises(BehaviorPinStoreError, match="cannot be empty"):
            pin_store.pin(
                domain_id="",
                agent_type="employee",
                context={},
                behavior=behavior_response,
            )

        with pytest.raises(BehaviorPinStoreError, match="invalid characters"):
            pin_store.pin(
                domain_id="invalid/domain",
                agent_type="employee",
                context={},
                behavior=behavior_response,
            )

    def test_pin_persists_to_disk(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test pin persists to disk file."""
        pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
            reason="persist test",
        )

        pin_file = pin_store._storage_dir / "shop_pins.json"
        assert pin_file.exists()

        with open(pin_file) as f:
            data = json.load(f)
        assert len(data["pins"]) == 1
        assert data["pins"][0]["reason"] == "persist test"


class TestBehaviorPinStoreUnpin:
    """Tests for unpin operation."""

    def test_unpin_removes_pinned_behavior(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test unpin removes a pinned behavior."""
        pin_id = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        result = pin_store.unpin(pin_id)

        assert result is True
        assert pin_store.total_pins == 0
        assert pin_store.get_by_id(pin_id) is None

    def test_unpin_nonexistent_returns_false(self, pin_store: BehaviorPinStore) -> None:
        """Test unpin returns False for nonexistent pin."""
        result = pin_store.unpin("pin-nonexistent")
        assert result is False

    def test_unpin_removes_from_disk(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test unpin removes entry from disk file."""
        pin_id = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        pin_store.unpin(pin_id)

        pin_file = pin_store._storage_dir / "shop_pins.json"
        # File should be removed when no pins remain
        assert not pin_file.exists()

    def test_unpin_preserves_other_pins(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test unpin only removes specified pin."""
        pin_id1 = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )
        pin_id2 = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "serve"},
            behavior=behavior_response,
        )

        pin_store.unpin(pin_id1)

        assert pin_store.total_pins == 1
        assert pin_store.get_by_id(pin_id2) is not None


class TestBehaviorPinStoreLookup:
    """Tests for lookup operations."""

    def test_get_by_id_returns_pinned(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test get_by_id returns correct pinned behavior."""
        pin_id = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        pinned = pin_store.get_by_id(pin_id)

        assert pinned is not None
        assert pinned.domain_id == "shop"
        assert pinned.agent_type == "employee"
        assert pinned.behavior.action == behavior_response.action

    def test_get_by_id_returns_none_for_nonexistent(self, pin_store: BehaviorPinStore) -> None:
        """Test get_by_id returns None for nonexistent pin."""
        result = pin_store.get_by_id("pin-nonexistent")
        assert result is None

    def test_get_by_key_returns_pinned(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test get_by_key returns correct pinned behavior."""
        pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        pinned = pin_store.get_by_key("shop", "employee", {"task": "wait"})

        assert pinned is not None
        assert pinned.behavior.action == behavior_response.action

    def test_get_by_key_returns_none_for_nonexistent(self, pin_store: BehaviorPinStore) -> None:
        """Test get_by_key returns None for nonexistent combination."""
        result = pin_store.get_by_key("shop", "employee", {"task": "wait"})
        assert result is None

    def test_get_behavior_returns_response(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test get_behavior returns just the BehaviorResponse."""
        pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        behavior = pin_store.get_behavior("shop", "employee", {"task": "wait"})

        assert behavior is not None
        assert behavior.action == behavior_response.action

    def test_is_pinned_returns_true_for_pinned(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test is_pinned returns True for pinned behavior."""
        pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        assert pin_store.is_pinned("shop", "employee", {"task": "wait"}) is True

    def test_is_pinned_returns_false_for_unpinned(self, pin_store: BehaviorPinStore) -> None:
        """Test is_pinned returns False for unpinned behavior."""
        assert pin_store.is_pinned("shop", "employee", {"task": "wait"}) is False


class TestBehaviorPinStoreList:
    """Tests for list operations."""

    def test_list_pins_returns_all(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test list_pins returns all pins."""
        pin_store.pin("shop", "employee", {"task": "wait"}, behavior_response)
        pin_store.pin("shop", "manager", {"task": "supervise"}, behavior_response)
        pin_store.pin("factory", "worker", {"task": "assemble"}, behavior_response)

        pins = pin_store.list_pins()

        assert len(pins) == 3

    def test_list_pins_filters_by_domain(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test list_pins filters by domain."""
        pin_store.pin("shop", "employee", {"task": "wait"}, behavior_response)
        pin_store.pin("shop", "manager", {"task": "supervise"}, behavior_response)
        pin_store.pin("factory", "worker", {"task": "assemble"}, behavior_response)

        shop_pins = pin_store.list_pins(domain_id="shop")

        assert len(shop_pins) == 2
        assert all(p.domain_id == "shop" for p in shop_pins)

    def test_list_pin_ids_returns_ids(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test list_pin_ids returns just the IDs."""
        pin_id1 = pin_store.pin("shop", "employee", {"task": "wait"}, behavior_response)
        pin_id2 = pin_store.pin("shop", "manager", {"task": "supervise"}, behavior_response)

        pin_ids = pin_store.list_pin_ids()

        assert len(pin_ids) == 2
        assert pin_id1 in pin_ids
        assert pin_id2 in pin_ids


class TestBehaviorPinStoreClear:
    """Tests for clear operations."""

    def test_clear_domain_removes_domain_pins(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_domain removes all pins for a domain."""
        pin_store.pin("shop", "employee", {"task": "wait"}, behavior_response)
        pin_store.pin("shop", "manager", {"task": "supervise"}, behavior_response)
        pin_store.pin("factory", "worker", {"task": "assemble"}, behavior_response)

        count = pin_store.clear_domain("shop")

        assert count == 2
        assert pin_store.total_pins == 1
        assert len(pin_store.list_pins(domain_id="shop")) == 0
        assert len(pin_store.list_pins(domain_id="factory")) == 1

    def test_clear_domain_removes_file(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_domain removes the domain's pin file."""
        pin_store.pin("shop", "employee", {"task": "wait"}, behavior_response)

        pin_store.clear_domain("shop")

        pin_file = pin_store._storage_dir / "shop_pins.json"
        assert not pin_file.exists()

    def test_clear_all_removes_all_pins(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_all removes all pins."""
        pin_store.pin("shop", "employee", {"task": "wait"}, behavior_response)
        pin_store.pin("factory", "worker", {"task": "assemble"}, behavior_response)

        count = pin_store.clear_all()

        assert count == 2
        assert pin_store.total_pins == 0


class TestBehaviorPinStorePersistence:
    """Tests for persistence across restarts."""

    def test_pins_persist_across_restart(
        self, tmp_path: Path, behavior_response: BehaviorResponse
    ) -> None:
        """Test pinned behavior persists across server restart (new store instance)."""
        pins_dir = tmp_path / "pins"

        # Create first store and pin behavior
        store1 = BehaviorPinStore(storage_dir=pins_dir)
        pin_id = store1.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
            reason="Persist test",
        )

        # Create second store (simulates restart)
        store2 = BehaviorPinStore(storage_dir=pins_dir)

        # Verify pin still exists
        assert store2.total_pins == 1
        pinned = store2.get_by_id(pin_id)
        assert pinned is not None
        assert pinned.behavior.action == behavior_response.action
        assert pinned.reason == "Persist test"

    def test_pins_survive_cache_clear_simulation(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test pinned behaviors are separate from cache (survive cache clear)."""
        pin_id = pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        # Even if we clear in-memory indices (simulating what cache.clear() does to cache)
        # The pin store maintains its own separate storage
        # The get methods should still work from the in-memory index

        # Verify pin is still accessible
        pinned = pin_store.get_by_id(pin_id)
        assert pinned is not None

        # And via key lookup
        behavior = pin_store.get_behavior("shop", "employee", {"task": "wait"})
        assert behavior is not None

    def test_pinned_behavior_takes_priority_simulation(self, pin_store: BehaviorPinStore) -> None:
        """Test that pinned behavior lookup is available for cache integration."""
        # Pin a specific behavior
        pinned_response = BehaviorResponse(
            action="pinned_action",
            parameters={"source": "pinned"},
            reasoning="This was pinned",
        )
        pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=pinned_response,
        )

        # Simulate cache lookup priority:
        # 1. Check pin store first
        pinned = pin_store.get_behavior("shop", "employee", {"task": "wait"})
        assert pinned is not None
        assert pinned.action == "pinned_action"

        # 2. For non-pinned, would fall through to cache/LLM
        not_pinned = pin_store.get_behavior("shop", "employee", {"task": "other"})
        assert not_pinned is None


class TestBehaviorPinStoreConcurrency:
    """Tests for concurrent access thread safety."""

    def test_concurrent_pin_operations(self, tmp_path: Path) -> None:
        """Test concurrent pin operations don't cause data corruption."""
        store = BehaviorPinStore(storage_dir=tmp_path / "pins")
        errors: list[Exception] = []
        num_threads = 10
        ops_per_thread = 20

        def worker(thread_id: int) -> None:
            try:
                for i in range(ops_per_thread):
                    behavior = BehaviorResponse(
                        action=f"action_{thread_id}_{i}",
                        parameters={},
                        reasoning="",
                    )
                    store.pin(
                        domain_id=f"domain{thread_id}",
                        agent_type="agent",
                        context={"index": i},
                        behavior=behavior,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # Each thread creates ops_per_thread pins
        assert store.total_pins == num_threads * ops_per_thread

    def test_concurrent_mixed_operations(self, tmp_path: Path) -> None:
        """Test concurrent mixed operations (pin, unpin, list)."""
        store = BehaviorPinStore(storage_dir=tmp_path / "pins")
        errors: list[Exception] = []

        # Pre-populate some pins
        pin_ids = []
        for i in range(10):
            behavior = BehaviorResponse(action=f"action_{i}", parameters={}, reasoning="")
            pin_id = store.pin("shop", "employee", {"index": i}, behavior)
            pin_ids.append(pin_id)

        def pinner(thread_id: int) -> None:
            try:
                for i in range(10):
                    behavior = BehaviorResponse(
                        action=f"new_{thread_id}_{i}",
                        parameters={},
                        reasoning="",
                    )
                    store.pin(f"domain{thread_id}", "agent", {"index": i}, behavior)
            except Exception as e:
                errors.append(e)

        def unpinner() -> None:
            try:
                for pin_id in pin_ids[:5]:
                    store.unpin(pin_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def lister() -> None:
            try:
                for _ in range(20):
                    _ = store.list_pins()
                    _ = store.get_stats()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(3):
                futures.append(executor.submit(pinner, i))
            futures.append(executor.submit(unpinner))
            futures.append(executor.submit(lister))

            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors occurred: {errors}"


class TestBehaviorPinStoreStats:
    """Tests for statistics."""

    def test_stats_tracks_counts(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test stats correctly tracks pin counts."""
        pin_store.pin("shop", "employee", {"task": "wait"}, behavior_response)
        pin_store.pin("shop", "manager", {"task": "supervise"}, behavior_response)
        pin_store.pin("factory", "worker", {"task": "assemble"}, behavior_response)

        stats = pin_store.get_stats()

        assert stats["total_pins"] == 3
        assert stats["domains_with_pins"] == 2
        assert stats["pins_by_domain"]["shop"] == 2
        assert stats["pins_by_domain"]["factory"] == 1

    def test_len_returns_total_pins(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test __len__ returns total pins."""
        assert len(pin_store) == 0

        pin_store.pin("shop", "employee", {"task": "wait"}, behavior_response)
        assert len(pin_store) == 1

        pin_store.pin("shop", "manager", {"task": "supervise"}, behavior_response)
        assert len(pin_store) == 2


class TestPinnedBehavior:
    """Tests for PinnedBehavior model."""

    def test_pinned_behavior_creation(self, behavior_response: BehaviorResponse) -> None:
        """Test PinnedBehavior can be created with required fields."""
        pinned = PinnedBehavior(
            pin_id="pin-test123",
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
            cache_key="shop:employee:abc123",
            pinned_at="2024-01-01T00:00:00+00:00",
            reason="Test pin",
        )

        assert pinned.pin_id == "pin-test123"
        assert pinned.domain_id == "shop"
        assert pinned.behavior.action == behavior_response.action

    def test_pinned_behavior_serialization(self, behavior_response: BehaviorResponse) -> None:
        """Test PinnedBehavior can be serialized and deserialized."""
        pinned = PinnedBehavior(
            pin_id="pin-test123",
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
            cache_key="shop:employee:abc123",
            pinned_at="2024-01-01T00:00:00+00:00",
            reason="Test pin",
        )

        # Serialize
        data = pinned.model_dump()
        assert isinstance(data, dict)

        # Deserialize
        restored = PinnedBehavior(**data)
        assert restored.pin_id == pinned.pin_id
        assert restored.behavior.action == pinned.behavior.action


class TestContextNormalization:
    """Tests for context normalization in pin lookups."""

    def test_same_context_different_order_matches(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test same context with different field order matches pin."""
        # Pin with one field order
        pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait", "location": "counter"},
            behavior=behavior_response,
        )

        # Lookup with different order
        pinned = pin_store.get_by_key("shop", "employee", {"location": "counter", "task": "wait"})

        assert pinned is not None
        assert pinned.behavior.action == behavior_response.action

    def test_context_with_timestamp_ignored(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test context timestamp fields are ignored in matching."""
        # Pin without timestamp
        pin_store.pin(
            domain_id="shop",
            agent_type="employee",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        # Lookup with timestamp (should be ignored)
        pinned = pin_store.get_by_key("shop", "employee", {"task": "wait", "timestamp": 12345})

        assert pinned is not None


class TestDomainIdValidation:
    """Tests for domain ID validation."""

    def test_empty_domain_id_raises(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test empty domain ID raises error."""
        with pytest.raises(BehaviorPinStoreError, match="cannot be empty"):
            pin_store.pin("", "employee", {}, behavior_response)

    def test_whitespace_domain_id_raises(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test whitespace-only domain ID raises error."""
        with pytest.raises(BehaviorPinStoreError, match="cannot be empty"):
            pin_store.pin("   ", "employee", {}, behavior_response)

    def test_path_traversal_domain_id_raises(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test path traversal in domain ID raises error."""
        with pytest.raises(BehaviorPinStoreError, match="invalid"):
            pin_store.pin("../evil", "employee", {}, behavior_response)

    def test_valid_domain_ids_accepted(
        self, pin_store: BehaviorPinStore, behavior_response: BehaviorResponse
    ) -> None:
        """Test valid domain IDs are accepted."""
        valid_ids = ["shop", "my_domain", "domain-1", "Domain123"]
        for domain_id in valid_ids:
            pin_id = pin_store.pin(domain_id, "employee", {}, behavior_response)
            assert pin_id.startswith("pin-")

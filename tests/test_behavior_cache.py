"""Tests for the BehaviorCache module."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from loopengine.behaviors.behavior_cache import BehaviorCache, CacheEntry
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
def cache() -> BehaviorCache:
    """Create a test cache with small size for testing eviction."""
    return BehaviorCache(max_size=5, default_ttl=60.0)


class TestBehaviorCacheInit:
    """Tests for BehaviorCache initialization."""

    def test_init_with_defaults(self) -> None:
        """Test cache initializes with default values."""
        with patch("loopengine.behaviors.behavior_cache.get_llm_config") as mock_config:
            mock_config.return_value.behavior_cache_ttl = 300
            cache = BehaviorCache()

            assert cache.max_size == BehaviorCache.DEFAULT_MAX_SIZE
            assert cache.default_ttl == 300.0
            assert cache.size == 0

    def test_init_with_custom_values(self) -> None:
        """Test cache initializes with custom values."""
        cache = BehaviorCache(max_size=100, default_ttl=120.0)

        assert cache.max_size == 100
        assert cache.default_ttl == 120.0

    def test_init_stats_start_at_zero(self, cache: BehaviorCache) -> None:
        """Test statistics are initialized to zero."""
        stats = cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["expirations"] == 0
        assert stats["size"] == 0


class TestBehaviorCacheSetGet:
    """Tests for set and get operations."""

    def test_set_then_get_returns_same_behavior(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test set then get returns the same behavior."""
        cache.set("key1", behavior_response)
        result = cache.get("key1")

        assert result is not None
        assert result.action == behavior_response.action
        assert result.parameters == behavior_response.parameters
        assert result.reasoning == behavior_response.reasoning

    def test_get_nonexistent_key_returns_none(self, cache: BehaviorCache) -> None:
        """Test get returns None for nonexistent key."""
        result = cache.get("nonexistent")

        assert result is None

    def test_get_increments_miss_counter(self, cache: BehaviorCache) -> None:
        """Test get increments miss counter for nonexistent key."""
        cache.get("nonexistent")

        assert cache.misses == 1
        assert cache.hits == 0

    def test_get_increments_hit_counter(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test get increments hit counter for existing key."""
        cache.set("key1", behavior_response)
        cache.get("key1")

        assert cache.hits == 1
        assert cache.misses == 0

    def test_set_overwrites_existing_key(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test set overwrites existing key."""
        cache.set("key1", behavior_response)

        new_response = BehaviorResponse(
            action="different_action",
            parameters={},
            reasoning="Different",
        )
        cache.set("key1", new_response)

        result = cache.get("key1")
        assert result is not None
        assert result.action == "different_action"
        assert cache.size == 1

    def test_set_with_custom_ttl(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test set with custom TTL."""
        cache.set("key1", behavior_response, ttl=0.1)

        # Should exist immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        assert cache.get("key1") is None


class TestBehaviorCacheTTL:
    """Tests for TTL expiration."""

    def test_get_after_ttl_expiry_returns_none(self, behavior_response: BehaviorResponse) -> None:
        """Test get returns None after TTL expires."""
        cache = BehaviorCache(max_size=10, default_ttl=0.1)
        cache.set("key1", behavior_response)

        # Should exist immediately
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        assert cache.get("key1") is None

    def test_expiration_increments_counter(self, behavior_response: BehaviorResponse) -> None:
        """Test expiration increments expiration counter."""
        cache = BehaviorCache(max_size=10, default_ttl=0.05)
        cache.set("key1", behavior_response)

        time.sleep(0.1)
        cache.get("key1")

        stats = cache.get_stats()
        assert stats["expirations"] == 1
        assert stats["misses"] == 1

    def test_cleanup_expired_removes_all_expired(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test cleanup_expired removes all expired entries."""
        # Add entries with very short TTL
        cache.set("key1", behavior_response, ttl=0.05)
        cache.set("key2", behavior_response, ttl=0.05)
        cache.set("key3", behavior_response, ttl=60.0)  # This one should survive

        time.sleep(0.1)
        removed = cache.cleanup_expired()

        assert removed == 2
        assert cache.size == 1
        assert "key3" in cache


class TestBehaviorCacheLRU:
    """Tests for LRU eviction."""

    def test_lru_eviction_when_max_size_reached(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test LRU eviction when max size is reached."""
        # Cache has max_size=5, fill it up
        for i in range(5):
            cache.set(f"key{i}", behavior_response)

        assert cache.size == 5

        # Add one more, should evict key0 (oldest)
        cache.set("key5", behavior_response)

        assert cache.size == 5
        assert cache.get("key0") is None  # Should be evicted
        assert cache.get("key5") is not None  # Should exist

    def test_lru_access_updates_order(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test accessing an entry moves it to most recently used."""
        # Fill cache
        for i in range(5):
            cache.set(f"key{i}", behavior_response)

        # Access key0 to make it most recently used
        cache.get("key0")

        # Add new entry, should evict key1 (now oldest)
        cache.set("key5", behavior_response)

        assert cache.get("key0") is not None  # Should still exist
        assert cache.get("key1") is None  # Should be evicted

    def test_eviction_increments_counter(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test eviction increments eviction counter."""
        # Fill cache and cause evictions
        for i in range(7):  # 5 max, 2 evictions
            cache.set(f"key{i}", behavior_response)

        stats = cache.get_stats()
        assert stats["evictions"] == 2


class TestBehaviorCacheConcurrency:
    """Tests for concurrent access thread safety."""

    def test_concurrent_access_no_data_corruption(
        self, behavior_response: BehaviorResponse
    ) -> None:
        """Test concurrent access doesn't cause data corruption."""
        cache = BehaviorCache(max_size=100, default_ttl=60.0)
        errors: list[Exception] = []
        num_threads = 10
        ops_per_thread = 100

        def worker(thread_id: int) -> None:
            try:
                for i in range(ops_per_thread):
                    key = f"thread{thread_id}:key{i}"
                    cache.set(key, behavior_response)
                    result = cache.get(key)
                    if result is not None:
                        assert result.action == behavior_response.action
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_mixed_operations(self, behavior_response: BehaviorResponse) -> None:
        """Test concurrent mixed operations (get, set, delete, stats)."""
        cache = BehaviorCache(max_size=50, default_ttl=60.0)
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(50):
                    cache.set(f"writer{thread_id}:key{i}", behavior_response)
            except Exception as e:
                errors.append(e)

        def reader(thread_id: int) -> None:
            try:
                for i in range(50):
                    cache.get(f"writer{thread_id % 3}:key{i}")
            except Exception as e:
                errors.append(e)

        def deleter() -> None:
            try:
                for i in range(20):
                    cache.delete(f"writer0:key{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def stats_checker() -> None:
            try:
                for _ in range(20):
                    stats = cache.get_stats()
                    assert "hits" in stats
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(3):
                futures.append(executor.submit(writer, i))
            for i in range(3):
                futures.append(executor.submit(reader, i))
            futures.append(executor.submit(deleter))
            futures.append(executor.submit(stats_checker))

            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_eviction_safety(self, behavior_response: BehaviorResponse) -> None:
        """Test concurrent operations during eviction don't cause issues."""
        cache = BehaviorCache(max_size=10, default_ttl=60.0)
        errors: list[Exception] = []

        def aggressive_writer(thread_id: int) -> None:
            try:
                for i in range(100):
                    cache.set(f"key{thread_id}_{i}", behavior_response)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=aggressive_writer, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have many evictions due to small cache size
        assert cache.get_stats()["evictions"] > 0


class TestBehaviorCacheStats:
    """Tests for cache statistics."""

    def test_stats_tracks_hits_misses(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test stats correctly tracks hits and misses."""
        cache.set("key1", behavior_response)

        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("nonexistent")  # miss
        cache.get("key1")  # hit

        stats = cache.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.75

    def test_reset_stats_clears_counters(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test reset_stats clears all counters."""
        cache.set("key1", behavior_response)
        cache.get("key1")
        cache.get("nonexistent")

        cache.reset_stats()
        stats = cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["expirations"] == 0

    def test_hit_rate_zero_when_no_requests(self, cache: BehaviorCache) -> None:
        """Test hit rate is 0 when no requests made."""
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0


class TestBehaviorCacheUtilities:
    """Tests for utility methods."""

    def test_delete_existing_key(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test delete returns True for existing key."""
        cache.set("key1", behavior_response)
        result = cache.delete("key1")

        assert result is True
        assert cache.get("key1") is None

    def test_delete_nonexistent_key(self, cache: BehaviorCache) -> None:
        """Test delete returns False for nonexistent key."""
        result = cache.delete("nonexistent")
        assert result is False

    def test_clear_removes_all_entries(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear removes all entries."""
        for i in range(3):
            cache.set(f"key{i}", behavior_response)

        count = cache.clear()

        assert count == 3
        assert cache.size == 0

    def test_contains_for_existing_key(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test __contains__ returns True for existing key."""
        cache.set("key1", behavior_response)
        assert "key1" in cache

    def test_contains_for_nonexistent_key(self, cache: BehaviorCache) -> None:
        """Test __contains__ returns False for nonexistent key."""
        assert "nonexistent" not in cache

    def test_contains_for_expired_key(self, behavior_response: BehaviorResponse) -> None:
        """Test __contains__ returns False for expired key."""
        cache = BehaviorCache(max_size=10, default_ttl=0.05)
        cache.set("key1", behavior_response)

        time.sleep(0.1)
        assert "key1" not in cache

    def test_len_returns_size(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test __len__ returns current size."""
        assert len(cache) == 0

        cache.set("key1", behavior_response)
        assert len(cache) == 1

        cache.set("key2", behavior_response)
        assert len(cache) == 2


class TestCacheInvalidation:
    """Tests for cache invalidation methods."""

    def test_clear_all_removes_all_entries(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_all removes all entries from the cache."""
        for i in range(3):
            cache.set(f"domain1:agent1:key{i}", behavior_response)
        cache.set("domain2:agent2:key1", behavior_response)

        count = cache.clear_all()

        assert count == 4
        assert cache.size == 0

    def test_clear_all_resets_stats(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_all resets all statistics counters."""
        cache.set("key1", behavior_response)
        cache.get("key1")  # hit
        cache.get("nonexistent")  # miss

        cache.clear_all()
        stats = cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["expirations"] == 0

    def test_clear_domain_removes_domain_entries(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_domain only removes that domain's cache entries."""
        # Add entries for multiple domains
        cache.set("domain1:agent1:hash1", behavior_response)
        cache.set("domain1:agent2:hash2", behavior_response)
        cache.set("domain2:agent1:hash3", behavior_response)
        cache.set("domain3:agent1:hash4", behavior_response)

        count = cache.clear_domain("domain1")

        assert count == 2
        assert cache.size == 2
        assert cache.get("domain1:agent1:hash1") is None
        assert cache.get("domain1:agent2:hash2") is None
        assert cache.get("domain2:agent1:hash3") is not None
        assert cache.get("domain3:agent1:hash4") is not None

    def test_clear_domain_with_no_matching_entries(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_domain returns 0 when no entries match."""
        cache.set("domain1:agent1:hash1", behavior_response)

        count = cache.clear_domain("nonexistent_domain")

        assert count == 0
        assert cache.size == 1

    def test_clear_domain_empty_cache(self, cache: BehaviorCache) -> None:
        """Test clear_domain on empty cache returns 0."""
        count = cache.clear_domain("any_domain")
        assert count == 0

    def test_clear_agent_type_removes_matching_entries(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_agent_type removes entries for specific domain and agent type."""
        # Add entries for different combinations
        cache.set("domain1:employee:hash1", behavior_response)
        cache.set("domain1:employee:hash2", behavior_response)
        cache.set("domain1:manager:hash3", behavior_response)
        cache.set("domain2:employee:hash4", behavior_response)

        count = cache.clear_agent_type("domain1", "employee")

        assert count == 2
        assert cache.size == 2
        assert cache.get("domain1:employee:hash1") is None
        assert cache.get("domain1:employee:hash2") is None
        assert cache.get("domain1:manager:hash3") is not None
        assert cache.get("domain2:employee:hash4") is not None

    def test_clear_agent_type_with_no_matching_entries(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_agent_type returns 0 when no entries match."""
        cache.set("domain1:agent1:hash1", behavior_response)

        count = cache.clear_agent_type("domain1", "nonexistent_agent")

        assert count == 0
        assert cache.size == 1

    def test_clear_agent_type_empty_cache(self, cache: BehaviorCache) -> None:
        """Test clear_agent_type on empty cache returns 0."""
        count = cache.clear_agent_type("domain", "agent")
        assert count == 0

    def test_ttl_expiry_triggers_automatic_removal(
        self, behavior_response: BehaviorResponse
    ) -> None:
        """Test TTL expiry triggers automatic removal during get()."""
        cache = BehaviorCache(max_size=10, default_ttl=0.05)
        cache.set("domain:agent:key", behavior_response)

        # Entry should exist immediately
        assert cache.get("domain:agent:key") is not None

        # Wait for expiry
        time.sleep(0.1)

        # Entry should be automatically removed on access
        assert cache.get("domain:agent:key") is None
        assert cache.get_stats()["expirations"] == 1

    def test_clear_all_preserves_cache_configuration(
        self, cache: BehaviorCache, behavior_response: BehaviorResponse
    ) -> None:
        """Test clear_all preserves max_size and default_ttl settings."""
        original_max_size = cache.max_size
        original_ttl = cache.default_ttl

        cache.set("key1", behavior_response)
        cache.clear_all()

        assert cache.max_size == original_max_size
        assert cache.default_ttl == original_ttl


class TestCacheAndPinStoreIntegration:
    """Integration tests for cache invalidation not affecting pinned behaviors."""

    def test_pinned_behaviors_survive_cache_clear_all(
        self, behavior_response: BehaviorResponse, tmp_path: Path
    ) -> None:
        """Test pinned behaviors are not affected by cache.clear_all()."""
        from loopengine.behaviors.behavior_pin_store import BehaviorPinStore

        cache = BehaviorCache(max_size=10, default_ttl=60.0)
        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        # Add entry to cache
        cache.set("domain:agent:hash", behavior_response)

        # Pin a behavior
        pin_id = pin_store.pin(
            domain_id="domain",
            agent_type="agent",
            context={"task": "wait"},
            behavior=behavior_response,
            reason="Test pin",
        )

        # Clear the cache
        cache.clear_all()

        # Cache should be empty
        assert cache.size == 0
        assert cache.get("domain:agent:hash") is None

        # Pinned behavior should still exist
        assert pin_store.get_by_id(pin_id) is not None
        pinned = pin_store.get_behavior("domain", "agent", {"task": "wait"})
        assert pinned is not None
        assert pinned.action == behavior_response.action

    def test_pinned_behaviors_survive_cache_clear_domain(
        self, behavior_response: BehaviorResponse, tmp_path: Path
    ) -> None:
        """Test pinned behaviors are not affected by cache.clear_domain()."""
        from loopengine.behaviors.behavior_pin_store import BehaviorPinStore

        cache = BehaviorCache(max_size=10, default_ttl=60.0)
        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        # Add entry to cache
        cache.set("domain:agent:hash", behavior_response)

        # Pin a behavior
        pin_id = pin_store.pin(
            domain_id="domain",
            agent_type="agent",
            context={"task": "wait"},
            behavior=behavior_response,
        )

        # Clear domain from cache
        cache.clear_domain("domain")

        # Cache entry should be gone
        assert cache.get("domain:agent:hash") is None

        # Pinned behavior should still exist
        assert pin_store.get_by_id(pin_id) is not None

    def test_unpin_removes_pinned_behavior(
        self, behavior_response: BehaviorResponse, tmp_path: Path
    ) -> None:
        """Test unpin removes the pinned behavior from pin store."""
        from loopengine.behaviors.behavior_pin_store import BehaviorPinStore

        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        # Pin a behavior
        pin_id = pin_store.pin(
            domain_id="domain",
            agent_type="agent",
            context={"task": "wait"},
            behavior=behavior_response,
        )
        assert pin_store.get_by_id(pin_id) is not None

        # Unpin
        result = pin_store.unpin(pin_id)

        assert result is True
        assert pin_store.get_by_id(pin_id) is None

    def test_list_pinned_returns_all_pins_for_domain(
        self, behavior_response: BehaviorResponse, tmp_path: Path
    ) -> None:
        """Test list_pins returns all pins for a specific domain."""
        from loopengine.behaviors.behavior_pin_store import BehaviorPinStore

        pin_store = BehaviorPinStore(storage_dir=tmp_path / "pins")

        # Pin behaviors for different domains
        pin_store.pin("domain1", "agent1", {"task": "a"}, behavior_response)
        pin_store.pin("domain1", "agent2", {"task": "b"}, behavior_response)
        pin_store.pin("domain2", "agent1", {"task": "c"}, behavior_response)

        # List pins for domain1
        domain1_pins = pin_store.list_pins("domain1")

        assert len(domain1_pins) == 2
        assert all(p.domain_id == "domain1" for p in domain1_pins)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self, behavior_response: BehaviorResponse) -> None:
        """Test CacheEntry can be created with required fields."""
        now = time.time()
        entry = CacheEntry(
            behavior=behavior_response,
            expires_at=now + 300,
            created_at=now,
        )

        assert entry.behavior == behavior_response
        assert entry.expires_at > now
        assert entry.created_at == now

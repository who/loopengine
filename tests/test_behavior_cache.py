"""Tests for the BehaviorCache module."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

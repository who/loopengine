"""In-memory cache for storing generated behaviors.

This module provides a thread-safe, TTL-based cache with LRU eviction
for storing generated behaviors to reduce API calls. Covers FR-008.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from loopengine.behaviors.config import get_llm_config
from loopengine.behaviors.llm_client import BehaviorResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached behavior entry with TTL tracking.

    Attributes:
        behavior: The cached BehaviorResponse.
        expires_at: Unix timestamp when this entry expires.
        created_at: Unix timestamp when this entry was created.
    """

    behavior: BehaviorResponse
    expires_at: float
    created_at: float


class BehaviorCache:
    """Thread-safe LRU cache for behavior responses with TTL expiration.

    Provides efficient caching of LLM-generated behaviors to reduce API calls
    for repeated or similar contexts. Memory-bounded with configurable max size
    and automatic LRU eviction.

    Thread-safe: All operations are protected by a reentrant lock for
    safe concurrent access from multiple threads.

    Example:
        >>> cache = BehaviorCache(max_size=1000, default_ttl=300)
        >>> response = BehaviorResponse(action="idle", parameters={}, reasoning="")
        >>> cache.set("agent:123:context_hash", response)
        >>> cached = cache.get("agent:123:context_hash")
        >>> print(cached.action)  # "idle"

    Environment Variables:
        BEHAVIOR_CACHE_TTL: Default TTL in seconds (default: 300)
        BEHAVIOR_CACHE_MAX_SIZE: Maximum cache entries (default: 1000)
    """

    DEFAULT_MAX_SIZE = 1000
    DEFAULT_TTL = 300  # 5 minutes

    def __init__(
        self,
        max_size: int | None = None,
        default_ttl: float | None = None,
    ) -> None:
        """Initialize the behavior cache.

        Args:
            max_size: Maximum number of entries to store. When exceeded, least
                recently used entries are evicted. Defaults to BEHAVIOR_CACHE_MAX_SIZE
                env var or 1000.
            default_ttl: Default time-to-live in seconds for cache entries.
                Defaults to BEHAVIOR_CACHE_TTL env var or 300 seconds.
        """
        config = get_llm_config()

        self._max_size = max_size if max_size is not None else self.DEFAULT_MAX_SIZE
        self._default_ttl = (
            default_ttl if default_ttl is not None else float(config.behavior_cache_ttl)
        )

        # OrderedDict maintains insertion order for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Lock for thread-safe access
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

        logger.info(
            "Initialized BehaviorCache with max_size=%d, default_ttl=%.1fs",
            self._max_size,
            self._default_ttl,
        )

    def get(self, key: str) -> BehaviorResponse | None:
        """Retrieve a cached behavior by key.

        Returns None if the key is not found or has expired.
        Accessing an entry moves it to the end (most recently used).

        Thread-safe.

        Args:
            key: The cache key to look up.

        Returns:
            The cached BehaviorResponse, or None if not found or expired.
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                logger.debug("Cache miss for key: %s", key)
                return None

            # Check if expired
            if time.time() > entry.expires_at:
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                self._expirations += 1
                logger.debug("Cache entry expired for key: %s", key)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug("Cache hit for key: %s", key)
            return entry.behavior

    def set(
        self,
        key: str,
        behavior: BehaviorResponse,
        ttl: float | None = None,
    ) -> None:
        """Store a behavior in the cache.

        If the cache is at capacity, evicts the least recently used entry.
        If the key already exists, updates the value and moves to most recently used.

        Thread-safe.

        Args:
            key: The cache key.
            behavior: The BehaviorResponse to cache.
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()

        entry = CacheEntry(
            behavior=behavior,
            expires_at=now + effective_ttl,
            created_at=now,
        )

        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                del self._cache[key]

            # Evict LRU entries if at capacity
            while len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug("Evicted LRU entry: %s", evicted_key)

            self._cache[key] = entry
            logger.debug("Cached behavior for key: %s (ttl=%.1fs)", key, effective_ttl)

    def delete(self, key: str) -> bool:
        """Remove a specific entry from the cache.

        Thread-safe.

        Args:
            key: The cache key to remove.

        Returns:
            True if the key was found and removed, False otherwise.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug("Deleted cache entry: %s", key)
                return True
            return False

    def clear(self) -> int:
        """Clear all entries from the cache.

        Thread-safe.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cleared %d entries from cache", count)
            return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        This is called automatically during get() for accessed entries,
        but can be called manually to proactively clean the cache.

        Thread-safe.

        Returns:
            Number of expired entries removed.
        """
        now = time.time()
        expired_keys: list[str] = []

        with self._lock:
            for key, entry in self._cache.items():
                if now > entry.expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._expirations += 1

            if expired_keys:
                logger.debug("Cleaned up %d expired entries", len(expired_keys))

            return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Thread-safe.

        Returns:
            Dict with hits, misses, hit_rate, size, max_size,
            evictions, expirations, and default_ttl.
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "size": len(self._cache),
                "max_size": self._max_size,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "default_ttl": self._default_ttl,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters to zero.

        Thread-safe.
        """
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0
            logger.debug("Reset cache statistics")

    @property
    def size(self) -> int:
        """Get the current number of entries in the cache. Thread-safe."""
        with self._lock:
            return len(self._cache)

    @property
    def max_size(self) -> int:
        """Get the maximum cache size."""
        return self._max_size

    @property
    def default_ttl(self) -> float:
        """Get the default TTL in seconds."""
        return self._default_ttl

    @property
    def hits(self) -> int:
        """Get the number of cache hits. Thread-safe."""
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        """Get the number of cache misses. Thread-safe."""
        with self._lock:
            return self._misses

    def __contains__(self, key: str) -> bool:
        """Check if a key exists and is not expired. Thread-safe."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if time.time() > entry.expires_at:
                return False
            return True

    def __len__(self) -> int:
        """Get the current number of entries. Thread-safe."""
        return self.size

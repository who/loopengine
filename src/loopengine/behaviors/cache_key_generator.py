"""Cache key generation for behavior caching.

This module provides consistent, deterministic cache key generation from
agent context for cache lookups. Keys are human-readable for debugging
and normalize context to be order-independent.
"""

import hashlib
import json
import re
from typing import Any

# Fields to exclude from cache key generation (non-significant/transient data)
DEFAULT_IGNORED_FIELDS: frozenset[str] = frozenset(
    {
        "timestamp",
        "timestamps",
        "created_at",
        "updated_at",
        "last_modified",
        "last_updated",
        "modified_at",
        "time",
        "datetime",
        "date",
        "current_time",
        "current_timestamp",
        "now",
        "tick",
        "tick_count",
        "frame",
        "frame_count",
        "_id",
        "_timestamp",
        "_created",
        "_updated",
    }
)

# Pattern to match timestamp-like field names
TIMESTAMP_PATTERN = re.compile(
    r"^(?:.*_)?(?:timestamp|time|datetime|date|tick|frame)(?:_.*)?$", re.IGNORECASE
)


def _is_timestamp_field(field_name: str) -> bool:
    """Check if a field name looks like a timestamp field.

    Args:
        field_name: The field name to check.

    Returns:
        True if the field appears to be a timestamp-related field.
    """
    return field_name.lower() in DEFAULT_IGNORED_FIELDS or bool(TIMESTAMP_PATTERN.match(field_name))


def _normalize_value(value: Any, ignored_fields: frozenset[str]) -> Any:
    """Recursively normalize a value for consistent hashing.

    Handles nested dicts and lists, filtering out ignored fields
    and sorting dict keys for deterministic output.

    Args:
        value: The value to normalize.
        ignored_fields: Set of field names to ignore.

    Returns:
        Normalized value suitable for JSON serialization.
    """
    if isinstance(value, dict):
        return _normalize_dict(value, ignored_fields)
    elif isinstance(value, (list, tuple)):
        return [_normalize_value(item, ignored_fields) for item in value]
    elif isinstance(value, (set, frozenset)):
        # Convert sets to sorted lists for determinism
        return sorted(_normalize_value(item, ignored_fields) for item in value)
    elif isinstance(value, float):
        # Round floats to avoid precision issues
        return round(value, 6)
    else:
        return value


def _normalize_dict(data: dict[str, Any], ignored_fields: frozenset[str]) -> dict[str, Any]:
    """Normalize a dictionary for consistent hashing.

    Removes ignored fields, normalizes nested values, and returns
    a dict that will serialize deterministically.

    Args:
        data: The dictionary to normalize.
        ignored_fields: Set of field names to ignore.

    Returns:
        Normalized dictionary.
    """
    result = {}
    for key in sorted(data.keys()):
        # Skip ignored fields
        if key in ignored_fields or _is_timestamp_field(key):
            continue
        result[key] = _normalize_value(data[key], ignored_fields)
    return result


class CacheKeyGenerator:
    """Generates consistent, deterministic cache keys from agent context.

    Keys are human-readable for debugging and include:
    - Domain ID
    - Agent type
    - Hash of normalized context

    Context normalization ensures:
    - Same inputs always produce the same key (deterministic)
    - Field order doesn't affect the key
    - Timestamp and transient fields are ignored

    Example:
        >>> generator = CacheKeyGenerator()
        >>> context = {"current_task": "idle", "inventory": ["bread", "cheese"]}
        >>> key = generator.generate_key("sandwich_shop", "employee", context)
        >>> print(key)  # "sandwich_shop:employee:a1b2c3d4"

        >>> # Same context in different order produces same key
        >>> context2 = {"inventory": ["bread", "cheese"], "current_task": "idle"}
        >>> key2 = generator.generate_key("sandwich_shop", "employee", context2)
        >>> assert key == key2

    Attributes:
        ignored_fields: Set of field names to exclude from key generation.
        hash_length: Length of the hash suffix in the key.
    """

    DEFAULT_HASH_LENGTH = 12

    def __init__(
        self,
        ignored_fields: frozenset[str] | None = None,
        hash_length: int = DEFAULT_HASH_LENGTH,
    ) -> None:
        """Initialize the cache key generator.

        Args:
            ignored_fields: Additional field names to ignore. These are
                added to the default ignored fields (timestamps, etc.).
            hash_length: Length of hash suffix in keys (default: 12).
        """
        if ignored_fields is not None:
            self._ignored_fields = DEFAULT_IGNORED_FIELDS | ignored_fields
        else:
            self._ignored_fields = DEFAULT_IGNORED_FIELDS

        self._hash_length = hash_length

    @property
    def ignored_fields(self) -> frozenset[str]:
        """Get the set of ignored field names."""
        return self._ignored_fields

    @property
    def hash_length(self) -> int:
        """Get the hash suffix length."""
        return self._hash_length

    def generate_key(
        self,
        domain_id: str,
        agent_type: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a cache key from domain, agent type, and context.

        The key format is: "{domain_id}:{agent_type}:{context_hash}"

        Args:
            domain_id: The domain identifier.
            agent_type: The type of agent.
            context: Optional context dictionary to include in the key.

        Returns:
            A human-readable cache key string.
        """
        # Normalize context for consistent hashing
        if context:
            normalized = _normalize_dict(context, self._ignored_fields)
        else:
            normalized = {}

        # Create deterministic JSON representation
        context_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

        # Generate hash
        context_hash = hashlib.sha256(context_json.encode("utf-8")).hexdigest()[: self._hash_length]

        # Build human-readable key
        return f"{domain_id}:{agent_type}:{context_hash}"

    def normalize_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """Normalize a context dictionary without generating a key.

        Useful for debugging or comparing contexts.

        Args:
            context: The context dictionary to normalize.

        Returns:
            Normalized context with ignored fields removed
            and consistent ordering.
        """
        return _normalize_dict(context, self._ignored_fields)

    def get_context_hash(self, context: dict[str, Any] | None) -> str:
        """Get just the hash portion of what would be the key.

        Args:
            context: The context dictionary to hash.

        Returns:
            The hash string (without domain/agent prefix).
        """
        if context:
            normalized = _normalize_dict(context, self._ignored_fields)
        else:
            normalized = {}

        context_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(context_json.encode("utf-8")).hexdigest()[: self._hash_length]


# Module-level default instance for convenience
_default_generator: CacheKeyGenerator | None = None


def get_default_generator() -> CacheKeyGenerator:
    """Get the default CacheKeyGenerator instance.

    Returns:
        The default generator (created on first call).
    """
    global _default_generator
    if _default_generator is None:
        _default_generator = CacheKeyGenerator()
    return _default_generator


def generate_key(
    domain_id: str,
    agent_type: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Generate a cache key using the default generator.

    Convenience function for simple use cases.

    Args:
        domain_id: The domain identifier.
        agent_type: The type of agent.
        context: Optional context dictionary.

    Returns:
        A human-readable cache key string.
    """
    return get_default_generator().generate_key(domain_id, agent_type, context)

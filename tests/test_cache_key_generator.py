"""Tests for the CacheKeyGenerator module."""

import pytest

from loopengine.behaviors.cache_key_generator import (
    DEFAULT_IGNORED_FIELDS,
    CacheKeyGenerator,
    _is_timestamp_field,
    _normalize_dict,
    _normalize_value,
    generate_key,
    get_default_generator,
)


class TestIsTimestampField:
    """Tests for timestamp field detection."""

    def test_exact_matches(self) -> None:
        """Test exact timestamp field names are detected."""
        assert _is_timestamp_field("timestamp")
        assert _is_timestamp_field("created_at")
        assert _is_timestamp_field("updated_at")
        assert _is_timestamp_field("tick")
        assert _is_timestamp_field("frame")

    def test_case_insensitive(self) -> None:
        """Test timestamp detection is case insensitive."""
        assert _is_timestamp_field("TIMESTAMP")
        assert _is_timestamp_field("Created_At")
        assert _is_timestamp_field("TICK")

    def test_pattern_matches(self) -> None:
        """Test pattern-based timestamp field detection."""
        assert _is_timestamp_field("start_time")
        assert _is_timestamp_field("end_timestamp")
        assert _is_timestamp_field("last_tick")
        assert _is_timestamp_field("frame_count")

    def test_non_timestamp_fields(self) -> None:
        """Test non-timestamp fields are not matched."""
        assert not _is_timestamp_field("name")
        assert not _is_timestamp_field("inventory")
        assert not _is_timestamp_field("current_task")
        assert not _is_timestamp_field("status")


class TestNormalizeValue:
    """Tests for value normalization."""

    def test_normalize_dict(self) -> None:
        """Test dict normalization removes ignored fields."""
        data = {"name": "test", "timestamp": 12345, "value": 1}
        result = _normalize_value(data, DEFAULT_IGNORED_FIELDS)

        assert result == {"name": "test", "value": 1}

    def test_normalize_list(self) -> None:
        """Test list normalization preserves order."""
        data = [1, 2, 3]
        result = _normalize_value(data, DEFAULT_IGNORED_FIELDS)

        assert result == [1, 2, 3]

    def test_normalize_nested_dict(self) -> None:
        """Test nested dict normalization."""
        data = {
            "outer": {
                "inner": "value",
                "timestamp": 12345,
            },
            "tick": 100,
        }
        result = _normalize_value(data, DEFAULT_IGNORED_FIELDS)

        assert result == {"outer": {"inner": "value"}}

    def test_normalize_set(self) -> None:
        """Test set is converted to sorted list."""
        data = {"c", "a", "b"}
        result = _normalize_value(data, DEFAULT_IGNORED_FIELDS)

        assert result == ["a", "b", "c"]

    def test_normalize_float(self) -> None:
        """Test float is rounded for precision."""
        data = 1.123456789
        result = _normalize_value(data, DEFAULT_IGNORED_FIELDS)

        assert result == 1.123457


class TestNormalizeDict:
    """Tests for dictionary normalization."""

    def test_removes_ignored_fields(self) -> None:
        """Test ignored fields are removed."""
        data = {
            "important": "value",
            "timestamp": 12345,
            "created_at": "2024-01-01",
        }
        result = _normalize_dict(data, DEFAULT_IGNORED_FIELDS)

        assert result == {"important": "value"}

    def test_order_independent(self) -> None:
        """Test field order doesn't affect result."""
        data1 = {"z": 1, "a": 2, "m": 3}
        data2 = {"a": 2, "m": 3, "z": 1}

        result1 = _normalize_dict(data1, DEFAULT_IGNORED_FIELDS)
        result2 = _normalize_dict(data2, DEFAULT_IGNORED_FIELDS)

        # Both should produce the same dict (keys sorted internally)
        assert list(result1.keys()) == ["a", "m", "z"]
        assert result1 == result2


class TestCacheKeyGenerator:
    """Tests for CacheKeyGenerator class."""

    @pytest.fixture
    def generator(self) -> CacheKeyGenerator:
        """Create a test generator."""
        return CacheKeyGenerator()

    def test_generate_key_basic(self, generator: CacheKeyGenerator) -> None:
        """Test basic key generation."""
        context = {"current_task": "idle"}
        key = generator.generate_key("sandwich_shop", "employee", context)

        assert key.startswith("sandwich_shop:employee:")
        assert len(key.split(":")) == 3
        # Hash portion should be 12 chars by default
        assert len(key.split(":")[2]) == 12

    def test_same_context_produces_same_key(self, generator: CacheKeyGenerator) -> None:
        """Test deterministic key generation."""
        context = {"current_task": "idle", "inventory": ["bread", "cheese"]}

        key1 = generator.generate_key("domain", "agent", context)
        key2 = generator.generate_key("domain", "agent", context)

        assert key1 == key2

    def test_different_field_order_produces_same_key(self, generator: CacheKeyGenerator) -> None:
        """Test order-independent key generation."""
        context1 = {"a": 1, "b": 2, "c": 3}
        context2 = {"c": 3, "a": 1, "b": 2}
        context3 = {"b": 2, "c": 3, "a": 1}

        key1 = generator.generate_key("domain", "agent", context1)
        key2 = generator.generate_key("domain", "agent", context2)
        key3 = generator.generate_key("domain", "agent", context3)

        assert key1 == key2 == key3

    def test_different_context_values_produce_different_key(
        self, generator: CacheKeyGenerator
    ) -> None:
        """Test different contexts produce different keys."""
        context1 = {"current_task": "idle"}
        context2 = {"current_task": "working"}

        key1 = generator.generate_key("domain", "agent", context1)
        key2 = generator.generate_key("domain", "agent", context2)

        assert key1 != key2

    def test_timestamp_ignored_in_key(self, generator: CacheKeyGenerator) -> None:
        """Test timestamp fields don't affect key."""
        context1 = {"current_task": "idle", "timestamp": 12345}
        context2 = {"current_task": "idle", "timestamp": 67890}
        context3 = {"current_task": "idle"}  # No timestamp

        key1 = generator.generate_key("domain", "agent", context1)
        key2 = generator.generate_key("domain", "agent", context2)
        key3 = generator.generate_key("domain", "agent", context3)

        assert key1 == key2 == key3

    def test_various_timestamp_fields_ignored(self, generator: CacheKeyGenerator) -> None:
        """Test various timestamp-like fields are ignored."""
        base_context = {"important": "data"}

        # All these should produce the same key
        contexts = [
            {**base_context},
            {**base_context, "timestamp": 1},
            {**base_context, "created_at": "2024"},
            {**base_context, "updated_at": "2024"},
            {**base_context, "tick": 100},
            {**base_context, "frame": 50},
            {**base_context, "current_time": 12345},
            {**base_context, "last_modified": "now"},
        ]

        keys = [generator.generate_key("d", "a", ctx) for ctx in contexts]

        assert all(k == keys[0] for k in keys)

    def test_key_is_human_readable(self, generator: CacheKeyGenerator) -> None:
        """Test key format is human readable."""
        key = generator.generate_key("sandwich_shop", "employee", {"task": "idle"})

        parts = key.split(":")
        assert parts[0] == "sandwich_shop"
        assert parts[1] == "employee"
        # Hash is lowercase hex
        assert parts[2].isalnum()
        assert parts[2].islower() or parts[2].isdigit()

    def test_empty_context(self, generator: CacheKeyGenerator) -> None:
        """Test key generation with empty context."""
        key1 = generator.generate_key("domain", "agent", {})
        key2 = generator.generate_key("domain", "agent", None)

        assert key1 == key2
        assert "domain:agent:" in key1

    def test_custom_ignored_fields(self) -> None:
        """Test custom ignored fields."""
        generator = CacheKeyGenerator(ignored_fields=frozenset({"custom_field", "another"}))

        context1 = {"important": "data", "custom_field": "ignored"}
        context2 = {"important": "data", "another": "also_ignored"}
        context3 = {"important": "data"}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)
        key3 = generator.generate_key("d", "a", context3)

        assert key1 == key2 == key3

    def test_custom_hash_length(self) -> None:
        """Test custom hash length."""
        generator = CacheKeyGenerator(hash_length=8)
        key = generator.generate_key("domain", "agent", {"task": "idle"})

        assert len(key.split(":")[2]) == 8

    def test_normalize_context_method(self, generator: CacheKeyGenerator) -> None:
        """Test normalize_context utility method."""
        context = {
            "important": "data",
            "timestamp": 12345,
            "nested": {"value": 1, "tick": 100},
        }

        normalized = generator.normalize_context(context)

        assert "timestamp" not in normalized
        assert normalized["important"] == "data"
        assert "tick" not in normalized["nested"]
        assert normalized["nested"]["value"] == 1

    def test_get_context_hash_method(self, generator: CacheKeyGenerator) -> None:
        """Test get_context_hash utility method."""
        context = {"task": "idle"}

        hash1 = generator.get_context_hash(context)
        hash2 = generator.get_context_hash(context)

        assert hash1 == hash2
        assert len(hash1) == generator.hash_length

    def test_ignored_fields_property(self, generator: CacheKeyGenerator) -> None:
        """Test ignored_fields property."""
        assert "timestamp" in generator.ignored_fields
        assert "created_at" in generator.ignored_fields

    def test_hash_length_property(self) -> None:
        """Test hash_length property."""
        generator = CacheKeyGenerator(hash_length=16)
        assert generator.hash_length == 16


class TestNestedContextNormalization:
    """Tests for nested context handling."""

    @pytest.fixture
    def generator(self) -> CacheKeyGenerator:
        """Create a test generator."""
        return CacheKeyGenerator()

    def test_nested_dict_order_independent(self, generator: CacheKeyGenerator) -> None:
        """Test nested dict field order doesn't matter."""
        context1 = {"outer": {"z": 1, "a": 2}}
        context2 = {"outer": {"a": 2, "z": 1}}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)

        assert key1 == key2

    def test_nested_timestamp_fields_ignored(self, generator: CacheKeyGenerator) -> None:
        """Test timestamp fields in nested dicts are ignored."""
        context1 = {"outer": {"value": 1, "timestamp": 100}}
        context2 = {"outer": {"value": 1, "timestamp": 200}}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)

        assert key1 == key2

    def test_list_of_dicts(self, generator: CacheKeyGenerator) -> None:
        """Test list of dicts is handled correctly."""
        context1 = {"items": [{"name": "a"}, {"name": "b"}]}
        context2 = {"items": [{"name": "a"}, {"name": "b"}]}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)

        assert key1 == key2

    def test_list_order_matters(self, generator: CacheKeyGenerator) -> None:
        """Test list element order does matter."""
        context1 = {"items": ["a", "b", "c"]}
        context2 = {"items": ["c", "b", "a"]}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)

        # List order should matter (different keys)
        assert key1 != key2

    def test_deeply_nested_structure(self, generator: CacheKeyGenerator) -> None:
        """Test deeply nested structure."""
        context = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep",
                        "timestamp": 12345,  # Should be ignored
                    }
                }
            }
        }

        key = generator.generate_key("d", "a", context)
        assert "d:a:" in key


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_default_generator_singleton(self) -> None:
        """Test default generator is singleton-like."""
        gen1 = get_default_generator()
        gen2 = get_default_generator()

        assert gen1 is gen2

    def test_generate_key_convenience(self) -> None:
        """Test module-level generate_key function."""
        key = generate_key("domain", "agent", {"task": "idle"})

        assert "domain:agent:" in key
        assert len(key.split(":")) == 3

    def test_generate_key_matches_instance(self) -> None:
        """Test module-level function matches instance method."""
        generator = CacheKeyGenerator()
        context = {"task": "idle", "inventory": ["item1"]}

        key1 = generate_key("domain", "agent", context)
        key2 = generator.generate_key("domain", "agent", context)

        assert key1 == key2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def generator(self) -> CacheKeyGenerator:
        """Create a test generator."""
        return CacheKeyGenerator()

    def test_empty_string_domain_and_agent(self, generator: CacheKeyGenerator) -> None:
        """Test empty string domain and agent type."""
        key = generator.generate_key("", "", {})
        assert key.startswith("::")

    def test_special_characters_in_domain(self, generator: CacheKeyGenerator) -> None:
        """Test special characters in domain/agent."""
        key = generator.generate_key("domain-with-dashes", "agent_with_underscores", {})
        assert "domain-with-dashes:agent_with_underscores:" in key

    def test_unicode_in_context(self, generator: CacheKeyGenerator) -> None:
        """Test unicode characters in context."""
        context = {"task": "日本語", "emoji": "🎉"}
        key1 = generator.generate_key("d", "a", context)
        key2 = generator.generate_key("d", "a", context)

        assert key1 == key2

    def test_none_values_in_context(self, generator: CacheKeyGenerator) -> None:
        """Test None values in context."""
        context = {"task": None, "value": 1}
        key = generator.generate_key("d", "a", context)

        assert "d:a:" in key

    def test_boolean_values(self, generator: CacheKeyGenerator) -> None:
        """Test boolean values in context."""
        context1 = {"flag": True}
        context2 = {"flag": False}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)

        assert key1 != key2

    def test_numeric_values(self, generator: CacheKeyGenerator) -> None:
        """Test various numeric values."""
        context1 = {"int": 1, "float": 1.5, "negative": -1}
        context2 = {"int": 1, "float": 1.5, "negative": -1}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)

        assert key1 == key2

    def test_float_precision_normalization(self, generator: CacheKeyGenerator) -> None:
        """Test floats with precision issues normalize correctly."""
        # These should produce the same key due to rounding
        context1 = {"value": 0.1 + 0.2}  # 0.30000000000000004
        context2 = {"value": 0.3}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)

        assert key1 == key2

    def test_tuple_in_context(self, generator: CacheKeyGenerator) -> None:
        """Test tuple values are handled like lists."""
        context1 = {"items": (1, 2, 3)}
        context2 = {"items": [1, 2, 3]}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)

        assert key1 == key2

    def test_frozenset_in_context(self, generator: CacheKeyGenerator) -> None:
        """Test frozenset values are converted to sorted lists."""
        context1 = {"items": frozenset([3, 1, 2])}
        context2 = {"items": frozenset([2, 3, 1])}

        key1 = generator.generate_key("d", "a", context1)
        key2 = generator.generate_key("d", "a", context2)

        # Should be same (sets are order-independent)
        assert key1 == key2

"""Tests for the domain configuration store."""

import json
from pathlib import Path

import pytest

from loopengine.behaviors.domain_parser import (
    AgentTypeSchema,
    DomainSchema,
    InteractionSchema,
    ResourceSchema,
)
from loopengine.behaviors.domain_store import (
    DomainMetadata,
    DomainStore,
    DomainStoreError,
    StoredDomain,
)


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    storage = tmp_path / "domains"
    storage.mkdir()
    return storage


@pytest.fixture
def domain_store(temp_storage_dir: Path) -> DomainStore:
    """Create a DomainStore with temporary storage."""
    return DomainStore(storage_dir=temp_storage_dir)


@pytest.fixture
def sample_schema() -> DomainSchema:
    """Create a sample domain schema."""
    return DomainSchema(
        domain_type="sandwich shop",
        description="A fast-food sandwich shop",
        agent_types=[
            AgentTypeSchema(
                name="sandwich_maker",
                role="Makes sandwiches",
                capabilities=["make_sandwich", "wrap_sandwich"],
            ),
            AgentTypeSchema(
                name="customer",
                role="Orders food",
                capabilities=["place_order", "pay"],
            ),
        ],
        resources=[
            ResourceSchema(name="bread", description="Sandwich bread", consumable=True),
            ResourceSchema(name="sandwich", description="Completed sandwich", consumable=True),
        ],
        interactions=[
            InteractionSchema(
                name="order",
                participants=["customer", "sandwich_maker"],
                description="Customer orders a sandwich",
            )
        ],
    )


@pytest.fixture
def minimal_schema() -> DomainSchema:
    """Create a minimal domain schema."""
    return DomainSchema(domain_type="minimal_domain")


class TestDomainMetadata:
    """Tests for the DomainMetadata model."""

    def test_metadata_basic(self) -> None:
        """Test DomainMetadata with basic data."""
        metadata = DomainMetadata(
            domain_id="test-domain",
            version=1,
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        assert metadata.domain_id == "test-domain"
        assert metadata.version == 1

    def test_metadata_default_version(self) -> None:
        """Test DomainMetadata has default version of 1."""
        metadata = DomainMetadata(
            domain_id="test",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        assert metadata.version == 1


class TestStoredDomain:
    """Tests for the StoredDomain model."""

    def test_stored_domain_basic(self, sample_schema: DomainSchema) -> None:
        """Test StoredDomain with basic data."""
        metadata = DomainMetadata(
            domain_id="shop",
            version=1,
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        stored = StoredDomain(metadata=metadata, schema=sample_schema)
        assert stored.metadata.domain_id == "shop"
        assert stored.schema_.domain_type == "sandwich shop"

    def test_stored_domain_serialization(self, sample_schema: DomainSchema) -> None:
        """Test StoredDomain serializes with correct alias."""
        metadata = DomainMetadata(
            domain_id="shop",
            version=1,
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        stored = StoredDomain(metadata=metadata, schema=sample_schema)
        data = stored.model_dump(by_alias=True)
        assert "schema" in data
        assert "schema_" not in data


class TestDomainStoreInit:
    """Tests for DomainStore initialization."""

    def test_init_with_default_dir(self) -> None:
        """Test initialization with default storage directory."""
        store = DomainStore()
        assert store._storage_dir == Path("data/domains")

    def test_init_with_custom_dir(self, temp_storage_dir: Path) -> None:
        """Test initialization with custom storage directory."""
        store = DomainStore(storage_dir=temp_storage_dir)
        assert store._storage_dir == temp_storage_dir

    def test_init_with_string_path(self, tmp_path: Path) -> None:
        """Test initialization with string path."""
        path_str = str(tmp_path / "custom_domains")
        store = DomainStore(storage_dir=path_str)
        assert store._storage_dir == Path(path_str)


class TestDomainStoreSave:
    """Tests for the save method."""

    def test_save_new_domain(self, domain_store: DomainStore, sample_schema: DomainSchema) -> None:
        """Test saving a new domain creates version 1."""
        result = domain_store.save("sandwich_shop", sample_schema)

        assert result.metadata.domain_id == "sandwich_shop"
        assert result.metadata.version == 1
        assert result.schema_.domain_type == "sandwich shop"
        assert result.metadata.created_at == result.metadata.updated_at

    def test_save_creates_file(
        self,
        domain_store: DomainStore,
        sample_schema: DomainSchema,
        temp_storage_dir: Path,
    ) -> None:
        """Test saving creates a JSON file."""
        domain_store.save("test_domain", sample_schema)

        file_path = temp_storage_dir / "test_domain.json"
        assert file_path.exists()

        with open(file_path) as f:
            data = json.load(f)
        assert data["metadata"]["domain_id"] == "test_domain"
        assert data["schema"]["domain_type"] == "sandwich shop"

    def test_save_update_increments_version(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test updating a domain increments the version."""
        first = domain_store.save("my_shop", sample_schema)
        assert first.metadata.version == 1

        updated_schema = DomainSchema(
            domain_type="updated shop",
            description="Updated description",
        )
        second = domain_store.save("my_shop", updated_schema)
        assert second.metadata.version == 2
        assert second.schema_.domain_type == "updated shop"

        # created_at should be preserved
        assert second.metadata.created_at == first.metadata.created_at
        # updated_at should change
        assert second.metadata.updated_at >= first.metadata.updated_at

    def test_save_multiple_updates(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test multiple updates increment version correctly."""
        for i in range(5):
            result = domain_store.save("versioned_domain", sample_schema)
            assert result.metadata.version == i + 1

    def test_save_empty_domain_id_raises_error(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test saving with empty domain_id raises DomainStoreError."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.save("", sample_schema)
        assert "cannot be empty" in str(exc_info.value)

    def test_save_whitespace_domain_id_raises_error(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test saving with whitespace domain_id raises DomainStoreError."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.save("   ", sample_schema)
        assert "cannot be empty" in str(exc_info.value)

    def test_save_invalid_chars_domain_id_raises_error(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test saving with invalid characters raises DomainStoreError."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.save("invalid/domain", sample_schema)
        assert "invalid characters" in str(exc_info.value)

    def test_save_path_traversal_raises_error(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test saving with path traversal attempt raises DomainStoreError."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.save("../sneaky", sample_schema)
        assert "invalid" in str(exc_info.value).lower()

    def test_save_with_hyphen_underscore_allowed(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test domain IDs with hyphens and underscores are allowed."""
        result = domain_store.save("my-domain_v1", sample_schema)
        assert result.metadata.domain_id == "my-domain_v1"

    def test_save_creates_storage_dir(self, tmp_path: Path, sample_schema: DomainSchema) -> None:
        """Test save creates storage directory if it doesn't exist."""
        new_dir = tmp_path / "new_domains"
        store = DomainStore(storage_dir=new_dir)

        assert not new_dir.exists()
        store.save("test", sample_schema)
        assert new_dir.exists()


class TestDomainStoreLoad:
    """Tests for the load method."""

    def test_load_existing_domain(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test loading an existing domain."""
        domain_store.save("shop", sample_schema)

        loaded = domain_store.load("shop")
        assert loaded.metadata.domain_id == "shop"
        assert loaded.metadata.version == 1
        assert loaded.schema_.domain_type == "sandwich shop"
        assert len(loaded.schema_.agent_types) == 2

    def test_load_nonexistent_domain_raises_error(self, domain_store: DomainStore) -> None:
        """Test loading non-existent domain raises DomainStoreError."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.load("nonexistent")
        assert "not found" in str(exc_info.value)

    def test_load_empty_domain_id_raises_error(self, domain_store: DomainStore) -> None:
        """Test loading with empty domain_id raises DomainStoreError."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.load("")
        assert "cannot be empty" in str(exc_info.value)

    def test_load_invalid_json_raises_error(
        self, domain_store: DomainStore, temp_storage_dir: Path
    ) -> None:
        """Test loading invalid JSON file raises DomainStoreError."""
        # Create an invalid JSON file
        invalid_path = temp_storage_dir / "invalid.json"
        invalid_path.write_text("not valid json {")

        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.load("invalid")
        assert "Invalid JSON" in str(exc_info.value)

    def test_load_preserves_full_schema(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test loading preserves all schema fields."""
        domain_store.save("full_schema", sample_schema)
        loaded = domain_store.load("full_schema")

        assert loaded.schema_.description == "A fast-food sandwich shop"
        assert len(loaded.schema_.agent_types) == 2
        assert len(loaded.schema_.resources) == 2
        assert len(loaded.schema_.interactions) == 1

        # Check nested data
        maker = next(a for a in loaded.schema_.agent_types if a.name == "sandwich_maker")
        assert "make_sandwich" in maker.capabilities


class TestDomainStoreExists:
    """Tests for the exists method."""

    def test_exists_returns_true_for_existing(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test exists returns True for existing domain."""
        domain_store.save("existing", sample_schema)
        assert domain_store.exists("existing") is True

    def test_exists_returns_false_for_nonexistent(self, domain_store: DomainStore) -> None:
        """Test exists returns False for non-existent domain."""
        assert domain_store.exists("nonexistent") is False

    def test_exists_returns_false_for_invalid_id(self, domain_store: DomainStore) -> None:
        """Test exists returns False for invalid domain ID (no exception)."""
        assert domain_store.exists("") is False
        assert domain_store.exists("invalid/id") is False


class TestDomainStoreDelete:
    """Tests for the delete method."""

    def test_delete_existing_domain(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test deleting an existing domain."""
        domain_store.save("to_delete", sample_schema)
        assert domain_store.exists("to_delete")

        domain_store.delete("to_delete")
        assert not domain_store.exists("to_delete")

    def test_delete_nonexistent_raises_error(self, domain_store: DomainStore) -> None:
        """Test deleting non-existent domain raises DomainStoreError."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.delete("nonexistent")
        assert "not found" in str(exc_info.value)

    def test_delete_empty_id_raises_error(self, domain_store: DomainStore) -> None:
        """Test deleting with empty ID raises DomainStoreError."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.delete("")
        assert "cannot be empty" in str(exc_info.value)


class TestDomainStoreList:
    """Tests for the list methods."""

    def test_list_domains_empty(self, domain_store: DomainStore) -> None:
        """Test list_domains returns empty list when no domains."""
        result = domain_store.list_domains()
        assert result == []

    def test_list_domains_returns_all(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test list_domains returns all stored domains."""
        domain_store.save("alpha", sample_schema)
        domain_store.save("beta", sample_schema)
        domain_store.save("gamma", sample_schema)

        result = domain_store.list_domains()
        assert len(result) == 3
        domain_ids = [m.domain_id for m in result]
        assert "alpha" in domain_ids
        assert "beta" in domain_ids
        assert "gamma" in domain_ids

    def test_list_domains_sorted(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test list_domains returns sorted by domain_id."""
        domain_store.save("zebra", sample_schema)
        domain_store.save("apple", sample_schema)
        domain_store.save("mango", sample_schema)

        result = domain_store.list_domains()
        domain_ids = [m.domain_id for m in result]
        assert domain_ids == ["apple", "mango", "zebra"]

    def test_list_domain_ids_empty(self, domain_store: DomainStore) -> None:
        """Test list_domain_ids returns empty list when no domains."""
        result = domain_store.list_domain_ids()
        assert result == []

    def test_list_domain_ids_returns_all(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test list_domain_ids returns all IDs."""
        domain_store.save("one", sample_schema)
        domain_store.save("two", sample_schema)

        result = domain_store.list_domain_ids()
        assert len(result) == 2
        assert "one" in result
        assert "two" in result

    def test_list_domain_ids_sorted(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test list_domain_ids returns sorted list."""
        domain_store.save("c_domain", sample_schema)
        domain_store.save("a_domain", sample_schema)
        domain_store.save("b_domain", sample_schema)

        result = domain_store.list_domain_ids()
        assert result == ["a_domain", "b_domain", "c_domain"]

    def test_list_handles_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test list methods handle non-existent storage directory."""
        store = DomainStore(storage_dir=tmp_path / "nonexistent")
        assert store.list_domains() == []
        assert store.list_domain_ids() == []


class TestDomainStoreGetVersion:
    """Tests for the get_version method."""

    def test_get_version_initial(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test get_version returns 1 for new domain."""
        domain_store.save("versioned", sample_schema)
        assert domain_store.get_version("versioned") == 1

    def test_get_version_after_updates(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test get_version returns correct version after updates."""
        domain_store.save("versioned", sample_schema)
        domain_store.save("versioned", sample_schema)
        domain_store.save("versioned", sample_schema)
        assert domain_store.get_version("versioned") == 3

    def test_get_version_nonexistent_raises_error(self, domain_store: DomainStore) -> None:
        """Test get_version raises error for non-existent domain."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.get_version("nonexistent")
        assert "not found" in str(exc_info.value)


class TestDomainStoreLoadWithMetadata:
    """Tests for the load_with_metadata method."""

    def test_load_with_metadata_returns_tuple(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test load_with_metadata returns (schema, metadata) tuple."""
        domain_store.save("test", sample_schema)

        schema, metadata = domain_store.load_with_metadata("test")

        assert isinstance(schema, DomainSchema)
        assert isinstance(metadata, dict)
        assert schema.domain_type == "sandwich shop"

    def test_load_with_metadata_includes_all_fields(
        self, domain_store: DomainStore, sample_schema: DomainSchema
    ) -> None:
        """Test load_with_metadata includes all metadata fields."""
        domain_store.save("test", sample_schema)

        _, metadata = domain_store.load_with_metadata("test")

        assert "domain_id" in metadata
        assert "version" in metadata
        assert "created_at" in metadata
        assert "updated_at" in metadata
        assert metadata["domain_id"] == "test"
        assert metadata["version"] == 1

    def test_load_with_metadata_nonexistent_raises_error(self, domain_store: DomainStore) -> None:
        """Test load_with_metadata raises error for non-existent domain."""
        with pytest.raises(DomainStoreError) as exc_info:
            domain_store.load_with_metadata("nonexistent")
        assert "not found" in str(exc_info.value)


class TestDomainStoreImport:
    """Tests for imports from behaviors package."""

    def test_import_from_behaviors_package(self) -> None:
        """Test all classes can be imported from behaviors package."""
        from loopengine.behaviors import (
            DomainMetadata,
            DomainStore,
            DomainStoreError,
            StoredDomain,
        )

        assert DomainStore is not None
        assert DomainStoreError is not None
        assert DomainMetadata is not None
        assert StoredDomain is not None


class TestDomainStoreRoundTrip:
    """Integration tests for save/load round trips."""

    def test_full_roundtrip(self, domain_store: DomainStore, sample_schema: DomainSchema) -> None:
        """Test full save, reload, update, verify cycle."""
        # Save initial
        first = domain_store.save("roundtrip", sample_schema)
        assert first.metadata.version == 1

        # Reload and verify
        loaded = domain_store.load("roundtrip")
        assert loaded.schema_.domain_type == sample_schema.domain_type
        assert len(loaded.schema_.agent_types) == len(sample_schema.agent_types)

        # Update
        updated_schema = DomainSchema(
            domain_type="updated shop",
            description="Changed description",
            agent_types=[
                AgentTypeSchema(name="new_agent", role="New role", capabilities=["new_action"])
            ],
        )
        second = domain_store.save("roundtrip", updated_schema)
        assert second.metadata.version == 2

        # Reload and verify update
        final = domain_store.load("roundtrip")
        assert final.schema_.domain_type == "updated shop"
        assert final.schema_.description == "Changed description"
        assert len(final.schema_.agent_types) == 1
        assert final.schema_.agent_types[0].name == "new_agent"

    def test_multiple_domains_isolated(
        self, domain_store: DomainStore, sample_schema: DomainSchema, minimal_schema: DomainSchema
    ) -> None:
        """Test multiple domains remain isolated."""
        domain_store.save("domain_a", sample_schema)
        domain_store.save("domain_b", minimal_schema)

        # Update only domain_a
        domain_store.save("domain_a", sample_schema)
        domain_store.save("domain_a", sample_schema)

        assert domain_store.get_version("domain_a") == 3
        assert domain_store.get_version("domain_b") == 1

        loaded_a = domain_store.load("domain_a")
        loaded_b = domain_store.load("domain_b")

        assert loaded_a.schema_.domain_type == "sandwich shop"
        assert loaded_b.schema_.domain_type == "minimal_domain"

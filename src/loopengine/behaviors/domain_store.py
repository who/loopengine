"""Persistence layer for domain configurations.

This module provides the DomainStore class for saving, loading, and managing
domain configurations as JSON files with versioning support.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loopengine.behaviors.domain_parser import DomainSchema

logger = logging.getLogger(__name__)


class DomainMetadata(BaseModel):
    """Metadata about a stored domain configuration.

    Attributes:
        domain_id: Unique identifier for the domain.
        version: Version number incremented on each update.
        created_at: ISO timestamp when domain was first created.
        updated_at: ISO timestamp of most recent update.
    """

    domain_id: str = Field(description="Unique identifier for the domain")
    version: int = Field(default=1, description="Version number")
    created_at: str = Field(description="ISO timestamp of creation")
    updated_at: str = Field(description="ISO timestamp of last update")


class StoredDomain(BaseModel):
    """A domain configuration with its metadata.

    Attributes:
        metadata: Domain metadata including ID and version.
        schema: The actual domain schema.
    """

    metadata: DomainMetadata = Field(description="Domain metadata")
    schema_: DomainSchema = Field(alias="schema", description="Domain schema")

    model_config = {"populate_by_name": True}


class DomainStoreError(Exception):
    """Exception raised when domain store operations fail."""

    pass


class DomainStore:
    """Persistence layer for domain configurations.

    Stores domain configurations as JSON files with automatic versioning.
    Each domain is stored in a separate file named by its domain_id.

    Example:
        >>> store = DomainStore()
        >>> store.save("my_shop", schema)  # First save, version 1
        >>> store.save("my_shop", updated_schema)  # Update, version 2
        >>> loaded = store.load("my_shop")
        >>> print(loaded.metadata.version)
        2
    """

    DEFAULT_STORAGE_DIR = "data/domains"

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        """Initialize the domain store.

        Args:
            storage_dir: Directory to store domain files. If not provided,
                uses DEFAULT_STORAGE_DIR relative to current working directory.
        """
        if storage_dir is None:
            self._storage_dir = Path(self.DEFAULT_STORAGE_DIR)
        else:
            self._storage_dir = Path(storage_dir)

    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_domain_path(self, domain_id: str) -> Path:
        """Get the file path for a domain ID.

        Args:
            domain_id: Unique identifier for the domain.

        Returns:
            Path to the domain's JSON file.
        """
        self._validate_domain_id(domain_id)
        return self._storage_dir / f"{domain_id}.json"

    def _validate_domain_id(self, domain_id: str) -> None:
        """Validate domain ID format.

        Args:
            domain_id: Domain ID to validate.

        Raises:
            DomainStoreError: If domain ID is invalid.
        """
        if not domain_id or not domain_id.strip():
            raise DomainStoreError("Domain ID cannot be empty")

        # Sanitize for filesystem safety - only allow alphanumeric, underscore, hyphen
        sanitized = domain_id.strip()
        if not all(c.isalnum() or c in ("_", "-") for c in sanitized):
            raise DomainStoreError(
                f"Domain ID '{domain_id}' contains invalid characters. "
                "Only alphanumeric characters, underscores, and hyphens are allowed."
            )

        # Prevent path traversal
        if ".." in sanitized or "/" in sanitized or "\\" in sanitized:
            raise DomainStoreError(f"Domain ID '{domain_id}' contains invalid path characters")

    def save(self, domain_id: str, schema: DomainSchema) -> StoredDomain:
        """Save a domain configuration.

        If the domain already exists, increments the version number.
        If it's new, creates it with version 1.

        Args:
            domain_id: Unique identifier for the domain.
            schema: Domain schema to save.

        Returns:
            StoredDomain with the saved schema and metadata.

        Raises:
            DomainStoreError: If save operation fails.
        """
        self._validate_domain_id(domain_id)
        self._ensure_storage_dir()

        domain_path = self._get_domain_path(domain_id)
        now = datetime.now(UTC).isoformat()

        # Check if domain exists to handle versioning
        existing: StoredDomain | None = None
        if domain_path.exists():
            try:
                existing = self._load_from_file(domain_path)
            except Exception:
                # If we can't load existing, treat as new
                logger.warning("Could not load existing domain %s, treating as new", domain_id)

        if existing:
            metadata = DomainMetadata(
                domain_id=domain_id,
                version=existing.metadata.version + 1,
                created_at=existing.metadata.created_at,
                updated_at=now,
            )
        else:
            metadata = DomainMetadata(
                domain_id=domain_id,
                version=1,
                created_at=now,
                updated_at=now,
            )

        stored = StoredDomain(metadata=metadata, schema=schema)

        try:
            with open(domain_path, "w", encoding="utf-8") as f:
                json.dump(stored.model_dump(by_alias=True), f, indent=2, ensure_ascii=False)
            logger.info("Saved domain %s version %d", domain_id, metadata.version)
            return stored
        except Exception as e:
            raise DomainStoreError(f"Failed to save domain '{domain_id}': {e}") from e

    def load(self, domain_id: str) -> StoredDomain:
        """Load a domain configuration.

        Args:
            domain_id: Unique identifier for the domain.

        Returns:
            StoredDomain with the schema and metadata.

        Raises:
            DomainStoreError: If domain doesn't exist or load fails.
        """
        self._validate_domain_id(domain_id)
        domain_path = self._get_domain_path(domain_id)

        if not domain_path.exists():
            raise DomainStoreError(f"Domain '{domain_id}' not found")

        return self._load_from_file(domain_path)

    def _load_from_file(self, path: Path) -> StoredDomain:
        """Load a StoredDomain from a file path.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed StoredDomain.

        Raises:
            DomainStoreError: If loading or parsing fails.
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return StoredDomain(**data)
        except json.JSONDecodeError as e:
            raise DomainStoreError(f"Invalid JSON in domain file '{path}': {e}") from e
        except Exception as e:
            raise DomainStoreError(f"Failed to load domain from '{path}': {e}") from e

    def exists(self, domain_id: str) -> bool:
        """Check if a domain exists.

        Args:
            domain_id: Unique identifier for the domain.

        Returns:
            True if domain exists, False otherwise.
        """
        try:
            self._validate_domain_id(domain_id)
            return self._get_domain_path(domain_id).exists()
        except DomainStoreError:
            return False

    def delete(self, domain_id: str) -> None:
        """Delete a domain configuration.

        Args:
            domain_id: Unique identifier for the domain.

        Raises:
            DomainStoreError: If domain doesn't exist or delete fails.
        """
        self._validate_domain_id(domain_id)
        domain_path = self._get_domain_path(domain_id)

        if not domain_path.exists():
            raise DomainStoreError(f"Domain '{domain_id}' not found")

        try:
            domain_path.unlink()
            logger.info("Deleted domain %s", domain_id)
        except Exception as e:
            raise DomainStoreError(f"Failed to delete domain '{domain_id}': {e}") from e

    def list_domains(self) -> list[DomainMetadata]:
        """List all available domains.

        Returns:
            List of DomainMetadata for all stored domains, sorted by domain_id.
        """
        if not self._storage_dir.exists():
            return []

        domains: list[DomainMetadata] = []
        for path in sorted(self._storage_dir.glob("*.json")):
            try:
                stored = self._load_from_file(path)
                domains.append(stored.metadata)
            except Exception as e:
                logger.warning("Could not load domain from %s: %s", path, e)

        return domains

    def list_domain_ids(self) -> list[str]:
        """List all available domain IDs.

        Returns:
            List of domain IDs, sorted alphabetically.
        """
        if not self._storage_dir.exists():
            return []

        domain_ids: list[str] = []
        for path in sorted(self._storage_dir.glob("*.json")):
            domain_ids.append(path.stem)

        return domain_ids

    def get_version(self, domain_id: str) -> int:
        """Get the current version of a domain.

        Args:
            domain_id: Unique identifier for the domain.

        Returns:
            Current version number.

        Raises:
            DomainStoreError: If domain doesn't exist.
        """
        stored = self.load(domain_id)
        return stored.metadata.version

    def load_with_metadata(self, domain_id: str) -> tuple[DomainSchema, dict[str, Any]]:
        """Load a domain and return schema with metadata dict.

        Convenience method for API responses.

        Args:
            domain_id: Unique identifier for the domain.

        Returns:
            Tuple of (DomainSchema, metadata dict).

        Raises:
            DomainStoreError: If domain doesn't exist.
        """
        stored = self.load(domain_id)
        metadata = {
            "domain_id": stored.metadata.domain_id,
            "version": stored.metadata.version,
            "created_at": stored.metadata.created_at,
            "updated_at": stored.metadata.updated_at,
        }
        return stored.schema_, metadata

"""Persistent storage for pinned behaviors.

This module provides the BehaviorPinStore class for saving, loading, and managing
pinned behaviors that persist across cache clears and server restarts. Covers FR-009.
"""

import json
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loopengine.behaviors.cache_key_generator import generate_key
from loopengine.behaviors.llm_client import BehaviorResponse

logger = logging.getLogger(__name__)


class PinnedBehavior(BaseModel):
    """A pinned behavior entry with metadata.

    Attributes:
        pin_id: Unique identifier for this pin.
        domain_id: Domain this behavior belongs to.
        agent_type: Type of agent this behavior is for.
        context: The agent context that triggers this behavior.
        behavior: The pinned BehaviorResponse.
        cache_key: Pre-computed cache key for fast lookups.
        pinned_at: ISO timestamp when this behavior was pinned.
        reason: Optional reason for pinning this behavior.
    """

    pin_id: str = Field(description="Unique identifier for this pin")
    domain_id: str = Field(description="Domain this behavior belongs to")
    agent_type: str = Field(description="Type of agent")
    context: dict[str, Any] = Field(description="Agent context that triggers this behavior")
    behavior: BehaviorResponse = Field(description="The pinned behavior")
    cache_key: str = Field(description="Pre-computed cache key for lookups")
    pinned_at: str = Field(description="ISO timestamp when pinned")
    reason: str = Field(default="", description="Optional reason for pinning")


class BehaviorPinStoreError(Exception):
    """Exception raised when pin store operations fail."""

    pass


class BehaviorPinStore:
    """Persistent storage for pinned behaviors.

    Stores pinned behaviors in JSON files, one per domain. Pinned behaviors
    are never evicted from cache and persist across server restarts.

    Thread-safe: All operations are protected by a reentrant lock.

    Example:
        >>> store = BehaviorPinStore()
        >>> behavior = BehaviorResponse(action="idle", parameters={}, reasoning="")
        >>> pin_id = store.pin("shop", "employee", {"task": "wait"}, behavior)
        >>> pinned = store.get_by_key("shop", "employee", {"task": "wait"})
        >>> print(pinned.behavior.action)  # "idle"
        >>> store.unpin(pin_id)

    File Structure:
        data/pins/{domain_id}.json - One file per domain containing all pins
    """

    DEFAULT_STORAGE_DIR = "data/pins"

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        """Initialize the pin store.

        Args:
            storage_dir: Directory to store pin files. If not provided,
                uses DEFAULT_STORAGE_DIR relative to current working directory.
        """
        if storage_dir is None:
            self._storage_dir = Path(self.DEFAULT_STORAGE_DIR)
        else:
            self._storage_dir = Path(storage_dir)

        self._lock = threading.RLock()

        # In-memory index for fast lookups: cache_key -> PinnedBehavior
        self._key_index: dict[str, PinnedBehavior] = {}

        # In-memory index: pin_id -> PinnedBehavior
        self._id_index: dict[str, PinnedBehavior] = {}

        # Load existing pins into memory
        self._load_all_pins()

    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_domain_path(self, domain_id: str) -> Path:
        """Get the file path for a domain's pins.

        Args:
            domain_id: Unique identifier for the domain.

        Returns:
            Path to the domain's pins JSON file.
        """
        self._validate_domain_id(domain_id)
        return self._storage_dir / f"{domain_id}_pins.json"

    def _validate_domain_id(self, domain_id: str) -> None:
        """Validate domain ID format.

        Args:
            domain_id: Domain ID to validate.

        Raises:
            BehaviorPinStoreError: If domain ID is invalid.
        """
        if not domain_id or not domain_id.strip():
            raise BehaviorPinStoreError("Domain ID cannot be empty")

        sanitized = domain_id.strip()
        if not all(c.isalnum() or c in ("_", "-") for c in sanitized):
            raise BehaviorPinStoreError(
                f"Domain ID '{domain_id}' contains invalid characters. "
                "Only alphanumeric characters, underscores, and hyphens are allowed."
            )

        if ".." in sanitized or "/" in sanitized or "\\" in sanitized:
            raise BehaviorPinStoreError(f"Domain ID '{domain_id}' contains invalid path characters")

    def _generate_pin_id(self) -> str:
        """Generate a unique pin ID.

        Returns:
            A unique pin ID string.
        """
        return f"pin-{uuid.uuid4().hex[:12]}"

    def _load_all_pins(self) -> None:
        """Load all pins from disk into memory indices."""
        if not self._storage_dir.exists():
            return

        for path in self._storage_dir.glob("*_pins.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                pins = [PinnedBehavior(**p) for p in data.get("pins", [])]
                for pin in pins:
                    self._key_index[pin.cache_key] = pin
                    self._id_index[pin.pin_id] = pin
                logger.debug("Loaded %d pins from %s", len(pins), path)
            except Exception as e:
                logger.warning("Could not load pins from %s: %s", path, e)

    def _save_domain_pins(self, domain_id: str) -> None:
        """Save all pins for a domain to disk.

        Args:
            domain_id: Domain whose pins to save.
        """
        self._ensure_storage_dir()
        domain_path = self._get_domain_path(domain_id)

        # Collect all pins for this domain
        domain_pins = [pin for pin in self._id_index.values() if pin.domain_id == domain_id]

        if not domain_pins:
            # Remove file if no pins remain
            if domain_path.exists():
                domain_path.unlink()
                logger.debug("Removed empty pin file for domain %s", domain_id)
            return

        data = {"pins": [pin.model_dump() for pin in domain_pins]}

        with open(domain_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug("Saved %d pins for domain %s", len(domain_pins), domain_id)

    def pin(
        self,
        domain_id: str,
        agent_type: str,
        context: dict[str, Any],
        behavior: BehaviorResponse,
        reason: str = "",
    ) -> str:
        """Pin a behavior so it is never regenerated or evicted.

        If a pin already exists for this domain/agent_type/context combination,
        it will be updated with the new behavior.

        Thread-safe.

        Args:
            domain_id: Domain this behavior belongs to.
            agent_type: Type of agent.
            context: Agent context that triggers this behavior.
            behavior: The behavior to pin.
            reason: Optional reason for pinning.

        Returns:
            The pin ID (either new or existing).

        Raises:
            BehaviorPinStoreError: If pin operation fails.
        """
        self._validate_domain_id(domain_id)
        cache_key = generate_key(domain_id, agent_type, context)

        with self._lock:
            # Check if already pinned for this key
            existing = self._key_index.get(cache_key)
            if existing:
                # Update existing pin
                updated = PinnedBehavior(
                    pin_id=existing.pin_id,
                    domain_id=domain_id,
                    agent_type=agent_type,
                    context=context,
                    behavior=behavior,
                    cache_key=cache_key,
                    pinned_at=existing.pinned_at,
                    reason=reason if reason else existing.reason,
                )
                self._key_index[cache_key] = updated
                self._id_index[updated.pin_id] = updated
                self._save_domain_pins(domain_id)
                logger.info(
                    "Updated pinned behavior %s for %s:%s", updated.pin_id, domain_id, agent_type
                )
                return updated.pin_id

            # Create new pin
            pin_id = self._generate_pin_id()
            now = datetime.now(UTC).isoformat()

            pinned = PinnedBehavior(
                pin_id=pin_id,
                domain_id=domain_id,
                agent_type=agent_type,
                context=context,
                behavior=behavior,
                cache_key=cache_key,
                pinned_at=now,
                reason=reason,
            )

            self._key_index[cache_key] = pinned
            self._id_index[pin_id] = pinned
            self._save_domain_pins(domain_id)

            logger.info("Pinned behavior %s for %s:%s", pin_id, domain_id, agent_type)
            return pin_id

    def unpin(self, pin_id: str) -> bool:
        """Remove a pinned behavior.

        Thread-safe.

        Args:
            pin_id: The pin ID to remove.

        Returns:
            True if the pin was found and removed, False otherwise.
        """
        with self._lock:
            pinned = self._id_index.get(pin_id)
            if pinned is None:
                logger.debug("Pin %s not found", pin_id)
                return False

            # Remove from indices
            del self._id_index[pin_id]
            if pinned.cache_key in self._key_index:
                del self._key_index[pinned.cache_key]

            # Persist changes
            self._save_domain_pins(pinned.domain_id)

            logger.info("Unpinned behavior %s", pin_id)
            return True

    def get_by_id(self, pin_id: str) -> PinnedBehavior | None:
        """Get a pinned behavior by its pin ID.

        Thread-safe.

        Args:
            pin_id: The pin ID to look up.

        Returns:
            The PinnedBehavior if found, None otherwise.
        """
        with self._lock:
            return self._id_index.get(pin_id)

    def get_by_key(
        self,
        domain_id: str,
        agent_type: str,
        context: dict[str, Any],
    ) -> PinnedBehavior | None:
        """Get a pinned behavior by domain, agent type, and context.

        Thread-safe.

        Args:
            domain_id: Domain to look up.
            agent_type: Agent type to look up.
            context: Context to match.

        Returns:
            The PinnedBehavior if found, None otherwise.
        """
        cache_key = generate_key(domain_id, agent_type, context)
        with self._lock:
            return self._key_index.get(cache_key)

    def get_behavior(
        self,
        domain_id: str,
        agent_type: str,
        context: dict[str, Any],
    ) -> BehaviorResponse | None:
        """Get just the behavior for a pinned entry.

        Convenience method for cache integration.

        Thread-safe.

        Args:
            domain_id: Domain to look up.
            agent_type: Agent type to look up.
            context: Context to match.

        Returns:
            The BehaviorResponse if pinned, None otherwise.
        """
        pinned = self.get_by_key(domain_id, agent_type, context)
        if pinned:
            return pinned.behavior
        return None

    def list_pins(self, domain_id: str | None = None) -> list[PinnedBehavior]:
        """List all pinned behaviors, optionally filtered by domain.

        Thread-safe.

        Args:
            domain_id: If provided, only return pins for this domain.

        Returns:
            List of PinnedBehavior objects.
        """
        with self._lock:
            if domain_id is None:
                return list(self._id_index.values())
            return [p for p in self._id_index.values() if p.domain_id == domain_id]

    def list_pin_ids(self, domain_id: str | None = None) -> list[str]:
        """List all pin IDs, optionally filtered by domain.

        Thread-safe.

        Args:
            domain_id: If provided, only return pin IDs for this domain.

        Returns:
            List of pin ID strings.
        """
        return [p.pin_id for p in self.list_pins(domain_id)]

    def is_pinned(
        self,
        domain_id: str,
        agent_type: str,
        context: dict[str, Any],
    ) -> bool:
        """Check if a behavior is pinned for the given parameters.

        Thread-safe.

        Args:
            domain_id: Domain to check.
            agent_type: Agent type to check.
            context: Context to check.

        Returns:
            True if pinned, False otherwise.
        """
        return self.get_by_key(domain_id, agent_type, context) is not None

    def clear_domain(self, domain_id: str) -> int:
        """Remove all pins for a domain.

        Thread-safe.

        Args:
            domain_id: Domain whose pins to clear.

        Returns:
            Number of pins removed.
        """
        self._validate_domain_id(domain_id)

        with self._lock:
            to_remove = [p for p in self._id_index.values() if p.domain_id == domain_id]
            for pin in to_remove:
                del self._id_index[pin.pin_id]
                if pin.cache_key in self._key_index:
                    del self._key_index[pin.cache_key]

            # Remove the file
            domain_path = self._get_domain_path(domain_id)
            if domain_path.exists():
                domain_path.unlink()

            logger.info("Cleared %d pins for domain %s", len(to_remove), domain_id)
            return len(to_remove)

    def clear_all(self) -> int:
        """Remove all pins from all domains.

        Thread-safe.

        Returns:
            Number of pins removed.
        """
        with self._lock:
            count = len(self._id_index)
            self._key_index.clear()
            self._id_index.clear()

            # Remove all pin files
            if self._storage_dir.exists():
                for path in self._storage_dir.glob("*_pins.json"):
                    path.unlink()

            logger.info("Cleared all %d pins", count)
            return count

    def get_stats(self) -> dict[str, Any]:
        """Get pin store statistics.

        Thread-safe.

        Returns:
            Dict with total_pins, domains_with_pins, pins_by_domain.
        """
        with self._lock:
            pins_by_domain: dict[str, int] = {}
            for pin in self._id_index.values():
                pins_by_domain[pin.domain_id] = pins_by_domain.get(pin.domain_id, 0) + 1

            return {
                "total_pins": len(self._id_index),
                "domains_with_pins": len(pins_by_domain),
                "pins_by_domain": pins_by_domain,
            }

    @property
    def total_pins(self) -> int:
        """Get the total number of pins. Thread-safe."""
        with self._lock:
            return len(self._id_index)

    def __len__(self) -> int:
        """Get the total number of pins. Thread-safe."""
        return self.total_pins

"""Periodic rediscovery triggers for automatic schema updates per PRD section 5.2.

This module implements trigger mechanisms for automatic genome schema rediscovery:
- System change detection (roles added/removed, links restructured)
- GA stagnation detection (fitness plateau across generations)
- Scheduled interval triggers (every N GA generations)
- Debouncing to prevent overlapping triggers

All triggers use a callback-based approach - when a trigger fires, it calls
the provided callback function with the current system description.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loopengine.model.world import World

logger = logging.getLogger(__name__)


class TriggerType(StrEnum):
    """Types of rediscovery triggers."""

    MANUAL = "manual"
    SYSTEM_CHANGE = "system_change"
    GA_STAGNATION = "ga_stagnation"
    SCHEDULED = "scheduled"


@dataclass
class TriggerEvent:
    """Record of a trigger firing event.

    Attributes:
        trigger_type: The type of trigger that fired.
        timestamp: When the trigger fired.
        metadata: Additional context about the trigger (e.g., stagnation details).
    """

    trigger_type: TriggerType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StagnationConfig:
    """Configuration for GA stagnation detection.

    Attributes:
        window_size: Number of generations to look back for stagnation detection.
        improvement_threshold: Minimum fitness improvement to not be considered stagnant.
            If best fitness improves by less than this over the window, it's stagnant.
    """

    window_size: int = 20
    improvement_threshold: float = 0.01


@dataclass
class ScheduledTriggerConfig:
    """Configuration for scheduled rediscovery triggers.

    Attributes:
        interval_generations: Trigger rediscovery every N GA generations.
        enabled: Whether scheduled triggers are enabled.
    """

    interval_generations: int = 100
    enabled: bool = True


class RediscoveryTriggerManager:
    """Manages periodic rediscovery triggers.

    This class monitors for conditions that should trigger genome schema
    rediscovery and invokes a callback when triggers fire. It implements
    debouncing to prevent multiple triggers from overlapping.

    Example:
        >>> def on_rediscovery(system_desc, trigger_type, metadata):
        ...     # Start rediscovery with system_desc
        ...     pass
        >>> manager = RediscoveryTriggerManager(callback=on_rediscovery)
        >>> manager.check_system_change(world)  # Check for system changes
        >>> manager.record_ga_generation(gen=10, best_fitness=0.85)  # Track GA
    """

    def __init__(
        self,
        callback: Callable[[dict[str, Any], TriggerType, dict[str, Any]], None] | None = None,
        stagnation_config: StagnationConfig | None = None,
        scheduled_config: ScheduledTriggerConfig | None = None,
        debounce_seconds: float = 5.0,
    ) -> None:
        """Initialize the trigger manager.

        Args:
            callback: Function called when a trigger fires.
                Signature: callback(system_description, trigger_type, metadata)
            stagnation_config: Configuration for GA stagnation detection.
            scheduled_config: Configuration for scheduled triggers.
            debounce_seconds: Minimum time between triggers (prevents overlapping).
        """
        self._callback = callback
        self._stagnation_config = stagnation_config or StagnationConfig()
        self._scheduled_config = scheduled_config or ScheduledTriggerConfig()
        self._debounce_seconds = debounce_seconds

        # State tracking
        self._lock = threading.Lock()
        self._last_trigger_time: datetime | None = None
        self._is_discovery_running = False
        self._trigger_history: list[TriggerEvent] = []

        # System change detection state
        self._last_system_hash: str | None = None

        # GA stagnation detection state
        self._fitness_history: list[tuple[int, float]] = []  # (generation, best_fitness)
        self._last_scheduled_generation: int = 0

    def set_callback(
        self, callback: Callable[[dict[str, Any], TriggerType, dict[str, Any]], None]
    ) -> None:
        """Set the callback function for when triggers fire.

        Args:
            callback: Function called when a trigger fires.
        """
        with self._lock:
            self._callback = callback

    def mark_discovery_started(self) -> None:
        """Mark that a discovery operation has started (for debouncing)."""
        with self._lock:
            self._is_discovery_running = True
            self._last_trigger_time = datetime.now()

    def mark_discovery_completed(self) -> None:
        """Mark that a discovery operation has completed."""
        with self._lock:
            self._is_discovery_running = False

    def is_trigger_allowed(self) -> bool:
        """Check if a trigger is allowed based on debouncing rules.

        Returns:
            True if a trigger can fire, False if debounced or discovery running.
        """
        with self._lock:
            # Don't trigger if discovery is already running
            if self._is_discovery_running:
                logger.debug("Trigger blocked: discovery already running")
                return False

            # Check debounce time
            if self._last_trigger_time is not None:
                elapsed = (datetime.now() - self._last_trigger_time).total_seconds()
                if elapsed < self._debounce_seconds:
                    logger.debug(
                        "Trigger blocked: debounce (%.1f < %.1f seconds)",
                        elapsed,
                        self._debounce_seconds,
                    )
                    return False

            return True

    def _fire_trigger(
        self,
        system_description: dict[str, Any],
        trigger_type: TriggerType,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Fire a trigger and invoke the callback.

        Args:
            system_description: System description for rediscovery.
            trigger_type: Type of trigger that fired.
            metadata: Additional context about the trigger.

        Returns:
            True if the trigger fired successfully, False if blocked.
        """
        metadata = metadata or {}

        # Check if trigger is allowed
        if not self.is_trigger_allowed():
            return False

        with self._lock:
            if self._callback is None:
                logger.warning("Trigger fired but no callback set")
                return False

            # Record the trigger
            event = TriggerEvent(
                trigger_type=trigger_type,
                timestamp=datetime.now(),
                metadata=metadata,
            )
            self._trigger_history.append(event)
            self._is_discovery_running = True
            self._last_trigger_time = event.timestamp

            logger.info(
                "Rediscovery trigger fired: %s (metadata: %s)",
                trigger_type.value,
                metadata,
            )

        # Call the callback outside the lock to avoid deadlocks
        try:
            self._callback(system_description, trigger_type, metadata)
            return True
        except Exception:
            logger.exception("Error in trigger callback")
            with self._lock:
                self._is_discovery_running = False
            return False

    def get_trigger_history(self) -> list[TriggerEvent]:
        """Get the history of trigger events.

        Returns:
            List of TriggerEvent objects in chronological order.
        """
        with self._lock:
            return list(self._trigger_history)

    # --- System Change Detection ---

    def _compute_system_hash(self, world: World) -> str:
        """Compute a hash of the system structure for change detection.

        The hash is based on:
        - Agent roles and their schemas
        - Link structure (source, dest, type)

        Args:
            world: The world to hash.

        Returns:
            A hex string hash of the system structure.
        """
        # Collect relevant system structure data
        structure: dict[str, Any] = {
            "roles": sorted({a.role for a in world.agents.values()}),
            "links": sorted(
                (link.source_id, link.dest_id, link.link_type.value)
                for link in world.links.values()
            ),
        }

        # Add schema versions if they exist
        if world.schemas:
            structure["schema_versions"] = {
                role: schema.version for role, schema in world.schemas.items()
            }

        # Compute hash
        json_str = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def check_system_change(self, world: World, system_description: dict[str, Any]) -> bool:
        """Check if the system structure has changed and trigger if so.

        Compares the current system structure hash with the last known hash.
        If they differ, fires a SYSTEM_CHANGE trigger.

        Args:
            world: The current world state.
            system_description: System description for rediscovery.

        Returns:
            True if a change was detected and trigger fired, False otherwise.
        """
        current_hash = self._compute_system_hash(world)

        with self._lock:
            if self._last_system_hash is None:
                # First check, just store the hash
                self._last_system_hash = current_hash
                logger.debug("System hash initialized: %s", current_hash)
                return False

            if current_hash == self._last_system_hash:
                # No change detected
                return False

            # System changed
            old_hash = self._last_system_hash
            self._last_system_hash = current_hash

        logger.info("System change detected (hash: %s -> %s)", old_hash, current_hash)

        return self._fire_trigger(
            system_description,
            TriggerType.SYSTEM_CHANGE,
            {"old_hash": old_hash, "new_hash": current_hash},
        )

    # --- GA Stagnation Detection ---

    def record_ga_generation(
        self,
        generation: int,
        best_fitness: float,
        system_description: dict[str, Any] | None = None,
    ) -> bool:
        """Record a GA generation's fitness for stagnation detection.

        Also checks for scheduled triggers based on generation count.

        Args:
            generation: The current generation number.
            best_fitness: The best fitness achieved this generation.
            system_description: System description for rediscovery (required for triggers).

        Returns:
            True if a trigger was fired (stagnation or scheduled), False otherwise.
        """
        with self._lock:
            # Record fitness
            self._fitness_history.append((generation, best_fitness))

            # Keep history bounded to window_size * 2 to avoid unbounded growth
            max_history = self._stagnation_config.window_size * 2
            if len(self._fitness_history) > max_history:
                self._fitness_history = self._fitness_history[-max_history:]

        if system_description is None:
            return False

        # Check for scheduled trigger
        if self._check_scheduled_trigger(generation, system_description):
            return True

        # Check for stagnation
        if self._check_stagnation(generation, system_description):
            return True

        return False

    def _check_scheduled_trigger(self, generation: int, system_description: dict[str, Any]) -> bool:
        """Check if a scheduled trigger should fire.

        Args:
            generation: Current generation number.
            system_description: System description for rediscovery.

        Returns:
            True if a scheduled trigger fired, False otherwise.
        """
        with self._lock:
            if not self._scheduled_config.enabled:
                return False

            interval = self._scheduled_config.interval_generations
            last_gen = self._last_scheduled_generation

            # Check if we've crossed an interval boundary
            if generation < interval:
                return False

            if generation // interval <= last_gen // interval:
                return False

            # Update last scheduled generation (do this in lock)
            self._last_scheduled_generation = generation

        logger.info(
            "Scheduled trigger at generation %d (interval: %d)",
            generation,
            interval,
        )

        return self._fire_trigger(
            system_description,
            TriggerType.SCHEDULED,
            {"generation": generation, "interval": interval},
        )

    def _check_stagnation(self, generation: int, system_description: dict[str, Any]) -> bool:
        """Check if GA has stagnated and trigger if so.

        Stagnation is detected when the best fitness improvement over the
        configured window is below the threshold.

        Args:
            generation: Current generation number.
            system_description: System description for rediscovery.

        Returns:
            True if stagnation detected and trigger fired, False otherwise.
        """
        with self._lock:
            window_size = self._stagnation_config.window_size
            threshold = self._stagnation_config.improvement_threshold

            # Need enough history to detect stagnation
            if len(self._fitness_history) < window_size:
                return False

            # Get fitness values from the window
            window = self._fitness_history[-window_size:]
            oldest_fitness = window[0][1]
            newest_fitness = window[-1][1]

            # Calculate improvement
            improvement = newest_fitness - oldest_fitness

        # Check if improvement is below threshold
        if improvement >= threshold:
            return False

        logger.info(
            "GA stagnation detected at generation %d "
            "(improvement: %.4f < %.4f over %d generations)",
            generation,
            improvement,
            threshold,
            window_size,
        )

        return self._fire_trigger(
            system_description,
            TriggerType.GA_STAGNATION,
            {
                "generation": generation,
                "window_size": window_size,
                "improvement": improvement,
                "threshold": threshold,
                "oldest_fitness": oldest_fitness,
                "newest_fitness": newest_fitness,
            },
        )

    # --- Configuration ---

    def update_stagnation_config(self, config: StagnationConfig) -> None:
        """Update the stagnation detection configuration.

        Args:
            config: New stagnation configuration.
        """
        with self._lock:
            self._stagnation_config = config

    def update_scheduled_config(self, config: ScheduledTriggerConfig) -> None:
        """Update the scheduled trigger configuration.

        Args:
            config: New scheduled trigger configuration.
        """
        with self._lock:
            self._scheduled_config = config

    def reset_state(self) -> None:
        """Reset all trigger state (useful for testing or restart)."""
        with self._lock:
            self._last_trigger_time = None
            self._is_discovery_running = False
            self._last_system_hash = None
            self._fitness_history = []
            self._last_scheduled_generation = 0
            # Note: we keep trigger_history for auditing


def extract_system_description(world: World) -> dict[str, Any]:
    """Extract a system description from a World for rediscovery.

    Converts the world's agent roles and link structure into the format
    expected by the Discoverer.

    Args:
        world: The world to extract from.

    Returns:
        A system description dict suitable for discover_schemas().
    """
    # Build roles from agents
    roles_by_name: dict[str, dict[str, Any]] = {}

    for agent in world.agents.values():
        if agent.role not in roles_by_name:
            roles_by_name[agent.role] = {
                "name": agent.role,
                "inputs": [],
                "outputs": [],
                "constraints": [],
                "links_to": [],
            }

    # Add link information
    for link in world.links.values():
        source_agent = world.agents.get(link.source_id)
        dest_agent = world.agents.get(link.dest_id)

        if source_agent and dest_agent:
            source_role = source_agent.role
            dest_role = dest_agent.role
            link_type = link.link_type.value

            # Add link_to entry for source role
            link_entry = f"{dest_role} ({link_type})"
            if link_entry not in roles_by_name[source_role]["links_to"]:
                roles_by_name[source_role]["links_to"].append(link_entry)

            # Add flow types as inputs/outputs if available
            flow_types = link.properties.get("flow_types", [])
            for flow_type in flow_types:
                if flow_type not in roles_by_name[dest_role]["inputs"]:
                    roles_by_name[dest_role]["inputs"].append(flow_type)
                if flow_type not in roles_by_name[source_role]["outputs"]:
                    roles_by_name[source_role]["outputs"].append(flow_type)

    # Add constraints from labels
    for agent in world.agents.values():
        role_desc = roles_by_name[agent.role]
        for label_name in agent.labels:
            label = world.labels.get(label_name)
            if label and label.context:
                for constraint in label.context.constraints:
                    if constraint not in role_desc["constraints"]:
                        role_desc["constraints"].append(constraint)

    return {
        "system": f"World with {len(world.agents)} agents",
        "roles": list(roles_by_name.values()),
    }

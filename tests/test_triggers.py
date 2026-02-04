"""Tests for periodic rediscovery triggers."""

import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from loopengine.corpora.sandwich_shop import create_world
from loopengine.discovery.triggers import (
    RediscoveryTriggerManager,
    ScheduledTriggerConfig,
    StagnationConfig,
    TriggerEvent,
    TriggerType,
    extract_system_description,
)
from loopengine.model import Agent, Link, LinkType, World


@pytest.fixture
def trigger_manager() -> RediscoveryTriggerManager:
    """Create a trigger manager with a mock callback."""
    callback = MagicMock()
    return RediscoveryTriggerManager(
        callback=callback,
        debounce_seconds=0.1,  # Short debounce for testing
    )


@pytest.fixture
def sample_world() -> World:
    """Create a sample world for testing."""
    return create_world()


@pytest.fixture
def sample_system_description() -> dict:
    """Create a sample system description."""
    return {
        "system": "Test system",
        "roles": [
            {"name": "owner", "inputs": [], "outputs": [], "constraints": []},
            {"name": "worker", "inputs": [], "outputs": [], "constraints": []},
        ],
    }


class TestTriggerManagerBasics:
    """Tests for basic trigger manager functionality."""

    def test_initialization(self, trigger_manager: RediscoveryTriggerManager) -> None:
        """Test trigger manager initializes correctly."""
        assert trigger_manager._callback is not None
        assert trigger_manager._debounce_seconds == 0.1
        assert trigger_manager._trigger_history == []
        assert not trigger_manager._is_discovery_running

    def test_set_callback(self) -> None:
        """Test setting callback after initialization."""
        manager = RediscoveryTriggerManager()
        assert manager._callback is None

        callback = MagicMock()
        manager.set_callback(callback)
        assert manager._callback is callback

    def test_mark_discovery_started(self, trigger_manager: RediscoveryTriggerManager) -> None:
        """Test marking discovery as started."""
        trigger_manager.mark_discovery_started()
        assert trigger_manager._is_discovery_running
        assert trigger_manager._last_trigger_time is not None

    def test_mark_discovery_completed(self, trigger_manager: RediscoveryTriggerManager) -> None:
        """Test marking discovery as completed."""
        trigger_manager.mark_discovery_started()
        trigger_manager.mark_discovery_completed()
        assert not trigger_manager._is_discovery_running

    def test_trigger_history(
        self,
        trigger_manager: RediscoveryTriggerManager,
        sample_system_description: dict,
    ) -> None:
        """Test trigger history tracking."""
        # Fire a manual trigger directly
        trigger_manager._fire_trigger(
            sample_system_description, TriggerType.MANUAL, {"test": "metadata"}
        )

        history = trigger_manager.get_trigger_history()
        assert len(history) == 1
        assert history[0].trigger_type == TriggerType.MANUAL
        assert history[0].metadata == {"test": "metadata"}
        assert isinstance(history[0].timestamp, datetime)

    def test_reset_state(
        self,
        trigger_manager: RediscoveryTriggerManager,
        sample_system_description: dict,
    ) -> None:
        """Test resetting trigger state."""
        # Set up some state
        trigger_manager._fire_trigger(sample_system_description, TriggerType.MANUAL)
        trigger_manager._fitness_history.append((1, 0.5))
        trigger_manager._last_system_hash = "test_hash"

        # Reset
        trigger_manager.reset_state()

        assert trigger_manager._last_trigger_time is None
        assert not trigger_manager._is_discovery_running
        assert trigger_manager._last_system_hash is None
        assert trigger_manager._fitness_history == []
        # History should be preserved for auditing
        assert len(trigger_manager._trigger_history) == 1


class TestDebouncing:
    """Tests for trigger debouncing."""

    def test_is_trigger_allowed_when_discovery_running(
        self, trigger_manager: RediscoveryTriggerManager
    ) -> None:
        """Test trigger blocked when discovery is running."""
        trigger_manager._is_discovery_running = True
        assert not trigger_manager.is_trigger_allowed()

    def test_is_trigger_allowed_within_debounce_window(
        self, trigger_manager: RediscoveryTriggerManager
    ) -> None:
        """Test trigger blocked within debounce window."""
        trigger_manager._last_trigger_time = datetime.now()
        assert not trigger_manager.is_trigger_allowed()

    def test_is_trigger_allowed_after_debounce_window(
        self, trigger_manager: RediscoveryTriggerManager
    ) -> None:
        """Test trigger allowed after debounce window."""
        trigger_manager._last_trigger_time = datetime.now()
        time.sleep(0.15)  # Wait longer than debounce_seconds
        assert trigger_manager.is_trigger_allowed()

    def test_multiple_triggers_debounced(
        self,
        trigger_manager: RediscoveryTriggerManager,
        sample_system_description: dict,
    ) -> None:
        """Test that multiple rapid triggers are debounced."""
        trigger_manager.mark_discovery_completed()  # Ensure not blocked by running

        # First trigger should fire
        result1 = trigger_manager._fire_trigger(sample_system_description, TriggerType.MANUAL)
        # Mark as completed so it's not blocked by running state
        trigger_manager.mark_discovery_completed()

        # Second trigger immediately after should be blocked
        result2 = trigger_manager._fire_trigger(sample_system_description, TriggerType.MANUAL)

        assert result1 is True
        assert result2 is False
        assert len(trigger_manager.get_trigger_history()) == 1


class TestSystemChangeDetection:
    """Tests for system change detection triggers."""

    def test_initial_hash_stored(
        self,
        trigger_manager: RediscoveryTriggerManager,
        sample_world: World,
        sample_system_description: dict,
    ) -> None:
        """Test initial system hash is stored without triggering."""
        result = trigger_manager.check_system_change(sample_world, sample_system_description)
        assert result is False
        assert trigger_manager._last_system_hash is not None

    def test_no_change_detected(
        self,
        trigger_manager: RediscoveryTriggerManager,
        sample_world: World,
        sample_system_description: dict,
    ) -> None:
        """Test no trigger when system unchanged."""
        # Initialize
        trigger_manager.check_system_change(sample_world, sample_system_description)

        # Check again - should not trigger
        result = trigger_manager.check_system_change(sample_world, sample_system_description)
        assert result is False
        assert len(trigger_manager.get_trigger_history()) == 0

    def test_change_detected_new_agent(
        self,
        trigger_manager: RediscoveryTriggerManager,
        sample_world: World,
        sample_system_description: dict,
    ) -> None:
        """Test trigger fires when agent added."""
        # Initialize
        trigger_manager.check_system_change(sample_world, sample_system_description)
        trigger_manager.mark_discovery_completed()  # Clear running state
        time.sleep(0.15)  # Wait for debounce

        # Add a new agent with a new role
        sample_world.agents["new_agent"] = Agent(
            id="new_agent",
            name="New",
            role="new_role",  # New role changes the hash
            genome={},
            labels=set(),
            loop_period=100,
        )

        result = trigger_manager.check_system_change(sample_world, sample_system_description)
        assert result is True
        history = trigger_manager.get_trigger_history()
        assert len(history) == 1
        assert history[0].trigger_type == TriggerType.SYSTEM_CHANGE

    def test_change_detected_new_link(
        self,
        trigger_manager: RediscoveryTriggerManager,
        sample_world: World,
        sample_system_description: dict,
    ) -> None:
        """Test trigger fires when link added."""
        # Initialize
        trigger_manager.check_system_change(sample_world, sample_system_description)
        trigger_manager.mark_discovery_completed()
        time.sleep(0.15)

        # Add a new link
        sample_world.links["new_link"] = Link(
            id="new_link",
            source_id="tom",
            dest_id="maria",
            link_type=LinkType.PEER,  # Different link type
        )

        result = trigger_manager.check_system_change(sample_world, sample_system_description)
        assert result is True

    def test_system_hash_deterministic(
        self, trigger_manager: RediscoveryTriggerManager, sample_world: World
    ) -> None:
        """Test system hash is deterministic."""
        hash1 = trigger_manager._compute_system_hash(sample_world)
        hash2 = trigger_manager._compute_system_hash(sample_world)
        assert hash1 == hash2


class TestGAStagnationDetection:
    """Tests for GA stagnation detection triggers."""

    def test_stagnation_config_defaults(self) -> None:
        """Test default stagnation configuration."""
        config = StagnationConfig()
        assert config.window_size == 20
        assert config.improvement_threshold == 0.01

    def test_record_ga_generation(self, trigger_manager: RediscoveryTriggerManager) -> None:
        """Test recording GA generations."""
        trigger_manager.record_ga_generation(1, 0.5)
        trigger_manager.record_ga_generation(2, 0.55)

        assert len(trigger_manager._fitness_history) == 2
        assert trigger_manager._fitness_history[0] == (1, 0.5)
        assert trigger_manager._fitness_history[1] == (2, 0.55)

    def test_no_stagnation_with_improvement(
        self,
        sample_system_description: dict,
    ) -> None:
        """Test no stagnation trigger when fitness is improving."""
        manager = RediscoveryTriggerManager(
            callback=MagicMock(),
            stagnation_config=StagnationConfig(window_size=5, improvement_threshold=0.01),
            debounce_seconds=0.0,
        )

        # Record improving fitness
        for gen in range(10):
            manager.record_ga_generation(gen, 0.5 + gen * 0.1, sample_system_description)
            manager.mark_discovery_completed()  # Reset running state

        # Should not have triggered stagnation (scheduled may have triggered)
        history = manager.get_trigger_history()
        stagnation_triggers = [e for e in history if e.trigger_type == TriggerType.GA_STAGNATION]
        assert len(stagnation_triggers) == 0

    def test_stagnation_detected(
        self,
        sample_system_description: dict,
    ) -> None:
        """Test stagnation trigger fires when fitness plateaus."""
        manager = RediscoveryTriggerManager(
            callback=MagicMock(),
            stagnation_config=StagnationConfig(window_size=5, improvement_threshold=0.01),
            scheduled_config=ScheduledTriggerConfig(enabled=False),  # Disable scheduled
            debounce_seconds=0.0,
        )

        # Record flat fitness
        for gen in range(10):
            manager.record_ga_generation(
                gen,
                0.5,
                sample_system_description,  # No improvement
            )
            manager.mark_discovery_completed()

        history = manager.get_trigger_history()
        stagnation_triggers = [e for e in history if e.trigger_type == TriggerType.GA_STAGNATION]
        assert len(stagnation_triggers) > 0
        assert "improvement" in stagnation_triggers[0].metadata

    def test_stagnation_not_triggered_with_insufficient_history(
        self,
        trigger_manager: RediscoveryTriggerManager,
        sample_system_description: dict,
    ) -> None:
        """Test stagnation not triggered before window is full."""
        trigger_manager._stagnation_config = StagnationConfig(
            window_size=20, improvement_threshold=0.01
        )
        trigger_manager._scheduled_config.enabled = False

        # Record only a few generations (less than window size)
        for gen in range(5):
            trigger_manager.record_ga_generation(gen, 0.5, sample_system_description)

        history = trigger_manager.get_trigger_history()
        stagnation_triggers = [e for e in history if e.trigger_type == TriggerType.GA_STAGNATION]
        assert len(stagnation_triggers) == 0

    def test_update_stagnation_config(self, trigger_manager: RediscoveryTriggerManager) -> None:
        """Test updating stagnation configuration."""
        new_config = StagnationConfig(window_size=50, improvement_threshold=0.05)
        trigger_manager.update_stagnation_config(new_config)

        assert trigger_manager._stagnation_config.window_size == 50
        assert trigger_manager._stagnation_config.improvement_threshold == 0.05


class TestScheduledTriggers:
    """Tests for scheduled interval triggers."""

    def test_scheduled_config_defaults(self) -> None:
        """Test default scheduled trigger configuration."""
        config = ScheduledTriggerConfig()
        assert config.interval_generations == 100
        assert config.enabled is True

    def test_scheduled_trigger_fires_at_interval(
        self,
        sample_system_description: dict,
    ) -> None:
        """Test scheduled trigger fires at configured interval."""
        manager = RediscoveryTriggerManager(
            callback=MagicMock(),
            scheduled_config=ScheduledTriggerConfig(interval_generations=10),
            debounce_seconds=0.0,
        )

        # Record generations up to and past the interval
        for gen in range(15):
            manager.record_ga_generation(gen, 0.5 + gen * 0.1, sample_system_description)
            manager.mark_discovery_completed()

        history = manager.get_trigger_history()
        scheduled_triggers = [e for e in history if e.trigger_type == TriggerType.SCHEDULED]
        assert len(scheduled_triggers) >= 1
        assert scheduled_triggers[0].metadata["generation"] == 10

    def test_scheduled_trigger_disabled(
        self,
        sample_system_description: dict,
    ) -> None:
        """Test scheduled trigger doesn't fire when disabled."""
        manager = RediscoveryTriggerManager(
            callback=MagicMock(),
            scheduled_config=ScheduledTriggerConfig(interval_generations=10, enabled=False),
            stagnation_config=StagnationConfig(window_size=100),  # Large window to avoid stagnation
            debounce_seconds=0.0,
        )

        for gen in range(15):
            manager.record_ga_generation(gen, 0.5 + gen * 0.1, sample_system_description)
            manager.mark_discovery_completed()

        history = manager.get_trigger_history()
        scheduled_triggers = [e for e in history if e.trigger_type == TriggerType.SCHEDULED]
        assert len(scheduled_triggers) == 0

    def test_update_scheduled_config(self, trigger_manager: RediscoveryTriggerManager) -> None:
        """Test updating scheduled trigger configuration."""
        new_config = ScheduledTriggerConfig(interval_generations=50, enabled=False)
        trigger_manager.update_scheduled_config(new_config)

        assert trigger_manager._scheduled_config.interval_generations == 50
        assert trigger_manager._scheduled_config.enabled is False


class TestExtractSystemDescription:
    """Tests for system description extraction."""

    def test_extract_basic_world(self, sample_world: World) -> None:
        """Test extracting system description from world."""
        desc = extract_system_description(sample_world)

        assert "system" in desc
        assert "roles" in desc
        assert len(desc["roles"]) > 0

    def test_extract_includes_all_roles(self, sample_world: World) -> None:
        """Test all roles are included in extraction."""
        desc = extract_system_description(sample_world)

        role_names = {r["name"] for r in desc["roles"]}
        expected_roles = {a.role for a in sample_world.agents.values()}
        assert role_names == expected_roles

    def test_extract_includes_link_info(self, sample_world: World) -> None:
        """Test link information is included."""
        desc = extract_system_description(sample_world)

        # Find the owner role
        owner_role = next(r for r in desc["roles"] if r["name"] == "owner")

        # Should have links_to other roles
        assert len(owner_role["links_to"]) > 0

    def test_extract_includes_flow_types(self, sample_world: World) -> None:
        """Test flow types are extracted as inputs/outputs."""
        desc = extract_system_description(sample_world)

        # Find the sandwich_maker role
        sandwich_maker = next(r for r in desc["roles"] if r["name"] == "sandwich_maker")

        # Should have some inputs or outputs from flow_types
        assert len(sandwich_maker["inputs"]) > 0 or len(sandwich_maker["outputs"]) > 0

    def test_extract_includes_constraints_from_labels(self, sample_world: World) -> None:
        """Test constraints are extracted from labels."""
        desc = extract_system_description(sample_world)

        # At least one role should have constraints
        all_constraints = []
        for role in desc["roles"]:
            all_constraints.extend(role["constraints"])
        assert len(all_constraints) > 0


class TestTriggerEventDataclass:
    """Tests for TriggerEvent dataclass."""

    def test_trigger_event_creation(self) -> None:
        """Test creating a trigger event."""
        event = TriggerEvent(
            trigger_type=TriggerType.MANUAL,
            metadata={"key": "value"},
        )

        assert event.trigger_type == TriggerType.MANUAL
        assert event.metadata == {"key": "value"}
        assert isinstance(event.timestamp, datetime)

    def test_trigger_event_default_values(self) -> None:
        """Test trigger event default values."""
        event = TriggerEvent(trigger_type=TriggerType.SCHEDULED)

        assert event.metadata == {}
        assert isinstance(event.timestamp, datetime)


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_trigger_types(self) -> None:
        """Test all trigger types are defined."""
        assert TriggerType.MANUAL == "manual"
        assert TriggerType.SYSTEM_CHANGE == "system_change"
        assert TriggerType.GA_STAGNATION == "ga_stagnation"
        assert TriggerType.SCHEDULED == "scheduled"


class TestCallbackInvocation:
    """Tests for callback invocation."""

    def test_callback_called_with_correct_args(
        self,
        sample_system_description: dict,
    ) -> None:
        """Test callback is called with correct arguments."""
        callback = MagicMock()
        manager = RediscoveryTriggerManager(callback=callback, debounce_seconds=0.0)

        manager._fire_trigger(
            sample_system_description,
            TriggerType.MANUAL,
            {"test": "data"},
        )

        callback.assert_called_once_with(
            sample_system_description,
            TriggerType.MANUAL,
            {"test": "data"},
        )

    def test_callback_exception_handled(
        self,
        sample_system_description: dict,
    ) -> None:
        """Test callback exception is handled gracefully."""
        callback = MagicMock(side_effect=Exception("Test error"))
        manager = RediscoveryTriggerManager(callback=callback, debounce_seconds=0.0)

        # Should not raise, but should return False
        result = manager._fire_trigger(sample_system_description, TriggerType.MANUAL)
        assert result is False
        assert not manager._is_discovery_running

    def test_no_callback_set(
        self,
        sample_system_description: dict,
    ) -> None:
        """Test trigger fails gracefully when no callback set."""
        manager = RediscoveryTriggerManager(debounce_seconds=0.0)

        result = manager._fire_trigger(sample_system_description, TriggerType.MANUAL)
        assert result is False

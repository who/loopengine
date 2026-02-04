"""Agent dataclass: Individual entity that senses, decides, and acts within a system."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loopengine.model.particle import Particle


class Phase(Enum):
    """OODA loop phases - the universal agent cycle."""

    SENSE = "sense"
    ORIENT = "orient"
    DECIDE = "decide"
    ACT = "act"


@dataclass
class Agent:
    """An individual entity that senses, decides, and acts within a system.

    Always represents a single individual, never a group or team.
    """

    # Identity
    id: str
    name: str
    role: str

    # Genome and labels
    genome: dict[str, float] = field(default_factory=dict)
    labels: set[str] = field(default_factory=set)

    # State
    internal_state: dict[str, Any] = field(default_factory=dict)
    input_buffer: list[Particle] = field(default_factory=list)
    output_buffer: list[Particle] = field(default_factory=list)

    # Loop configuration
    loop_period: int = 60  # ticks per full OODA revolution
    loop_phase: Phase = Phase.SENSE
    phase_tick: int = 0

    # Policy: callable(sensed_inputs, genome, internal_state) -> list[Particle]
    policy: Callable[[list[Particle], dict[str, float], dict[str, Any]], list[Particle]] | None = (
        None
    )

    # Position (for force-directed layout)
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

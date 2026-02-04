"""World and ExternalInput dataclasses per PRD section 3.6."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loopengine.model.agent import Agent
    from loopengine.model.genome import GenomeSchema
    from loopengine.model.label import Label
    from loopengine.model.link import Link
    from loopengine.model.particle import Particle


@dataclass
class ExternalInput:
    """Configuration for external input generation (e.g., customer arrivals).

    External inputs spawn particles at configured rates, simulating events
    from outside the system like customer arrivals or supply deliveries.
    """

    name: str  # e.g. "customer_arrivals"
    target_agent_id: str  # who receives the input
    rate: float  # average particles per tick
    variance: float = 0.0  # randomness in arrival rate
    particle_type: str = ""  # what kind of particle is generated
    payload_generator: Callable[[], dict[str, Any]] = field(
        default_factory=lambda: lambda: {}
    )  # function that creates particle payloads
    schedule: Callable[[int], float] = field(
        default_factory=lambda: lambda tick: 1.0
    )  # function(tick) → rate_multiplier (for rush patterns)


@dataclass
class World:
    """Container holding all simulation state.

    The World is the source of truth for the simulation, containing all agents,
    links, particles, labels, and schemas.
    """

    # Entity containers keyed by id/name
    agents: dict[str, Agent] = field(default_factory=dict)  # id → Agent
    links: dict[str, Link] = field(default_factory=dict)  # id → Link
    particles: dict[str, Particle] = field(default_factory=dict)  # id → Particle (active only)
    labels: dict[str, Label] = field(default_factory=dict)  # name → Label
    schemas: dict[str, GenomeSchema] = field(default_factory=dict)  # role → GenomeSchema

    # Simulation clock
    tick: int = 0  # global tick counter
    time: float = 0.0  # elapsed simulation time
    speed: float = 1.0  # ticks per second (simulation speed)

    # External input configuration
    external_inputs: list[ExternalInput] = field(default_factory=list)

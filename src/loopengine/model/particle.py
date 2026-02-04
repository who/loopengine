"""Particle dataclass: Discrete unit of flow traveling along a link between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Particle:
    """A discrete unit of flow traveling along a link from one agent to another.

    Particles represent orders, products, directives, reports, etc.
    When progress >= 1.0, the particle is delivered to the destination agent's input buffer.
    """

    id: str
    particle_type: str  # "order_ticket", "sandwich", "directive", etc.
    payload: dict[str, Any] = field(default_factory=dict)

    # Routing
    source_id: str = ""  # originating agent id
    dest_id: str = ""  # destination agent id
    link_id: str = ""  # which link it's traveling on

    # Transit progress
    progress: float = 0.0  # 0.0 (at source) to 1.0 (at destination)
    speed: float = 0.1  # progress increment per tick

    # Lifecycle
    alive: bool = True  # set to False when delivered or expired

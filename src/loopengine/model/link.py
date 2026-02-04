"""Link dataclass: Typed, directional connection between two agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LinkType(Enum):
    """Types of relationships between agents."""

    HIERARCHICAL = "hierarchical"  # Authority relationship (supervisor → worker)
    PEER = "peer"  # Coordination between equals
    SERVICE = "service"  # Service provider/consumer relationship
    COMPETITIVE = "competitive"  # Competitive relationship


@dataclass
class Link:
    """A typed, directional connection between two agents.

    Links carry properties defining the relationship (authority, autonomy, flow types).
    Bidirectional relationships are modeled as two separate Link objects.
    """

    id: str
    source_id: str  # agent id
    dest_id: str  # agent id
    link_type: LinkType

    # Properties may include:
    #   authority_scope: list[str]
    #   autonomy_granted: float
    #   fitness_definition: list[str]
    #   resource_control: list[str]
    #   flow_types: list[str]      # what particle types travel this link
    #   bandwidth: float           # how many particles per tick can traverse
    #   latency: int               # ticks for a particle to traverse
    properties: dict[str, Any] = field(default_factory=dict)

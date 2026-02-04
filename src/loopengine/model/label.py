"""Label and LabelContext dataclasses: Shared context among agents."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LabelContext:
    """Context information shared by agents carrying a label.

    Labels provide shared constraints, resources, and norms without acting themselves.
    """

    constraints: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    norms: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Label:
    """A shared identifier carried by agents who share context.

    Labels don't reference agents directly - agents carry label names in their
    labels set. To find all agents with a given label, query the World's agent
    collection.
    """

    name: str
    context: LabelContext = field(default_factory=LabelContext)

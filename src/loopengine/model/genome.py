"""GenomeSchema and GenomeTrait dataclasses per PRD section 3.5."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GenomeTrait:
    """A single trait within a genome schema.

    Defines a dimension along which an agent can vary that affects performance.
    """

    name: str
    description: str
    min_val: float = 0.0
    max_val: float = 1.0
    category: str = ""  # physical, cognitive, social, temperamental, skill
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class GenomeSchema:
    """AI-discovered definition of meaningful traits for a role.

    The GenomeSchema is the AI's output. It defines what traits are meaningful
    for a given role. Individual agent genomes are instances of this schema,
    but loosely coupled — genomes may have extra or missing keys relative to
    the current schema.
    """

    role: str
    traits: dict[str, GenomeTrait] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)
    discovery_prompt: str = ""
    version: int = 1

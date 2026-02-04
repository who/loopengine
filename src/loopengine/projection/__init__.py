"""Frame projection: World state to visual frame snapshots."""

from loopengine.projection.projector import (
    AgentVisual,
    Frame,
    LabelRegionVisual,
    LinkVisual,
    ParticleVisual,
    compute_breathing_radius,
    project,
)

__all__ = [
    "AgentVisual",
    "Frame",
    "LabelRegionVisual",
    "LinkVisual",
    "ParticleVisual",
    "compute_breathing_radius",
    "project",
]

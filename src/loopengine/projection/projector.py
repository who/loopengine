"""Frame projector: World state to visual Frame for rendering.

Converts World simulation state into Frame dataclasses suitable for frontend rendering.
Each Frame is a snapshot containing all visual elements needed to render one animation frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loopengine.model.agent import Agent
    from loopengine.model.link import Link
    from loopengine.model.particle import Particle
    from loopengine.model.world import World


@dataclass
class AgentVisual:
    """Visual representation of an agent for rendering.

    Agents display as breathing amoeba shapes with glow indicating load.
    """

    id: str
    name: str
    role: str

    # Position from force-directed layout
    x: float
    y: float

    # Visual properties
    radius: float = 20.0  # Base radius
    breathing_phase: float = 0.0  # Current phase in breathing cycle (0 to 2π)
    breathing_rate: float = 0.05  # Phase increment per tick
    color: str = "#4a90d9"  # Fill color
    glow_intensity: float = 0.0  # 0.0 to 1.0, based on input buffer depth

    # State indicators
    ooda_phase: str = "sense"  # Current OODA phase
    labels: list[str] = field(default_factory=list)


@dataclass
class LinkVisual:
    """Visual representation of a link for rendering.

    Links render as swaying bezier curves with thickness encoding density.
    """

    id: str
    source_id: str
    dest_id: str
    link_type: str  # "hierarchical", "peer", "service", "competitive"

    # Bezier curve control points: [start, control1, control2, end]
    control_points: list[tuple[float, float]] = field(default_factory=list)

    # Visual properties
    thickness: float = 2.0  # Base thickness (scaled by recent traffic)
    sway_phase: float = 0.0  # Current phase in sway animation
    color: str = "#666666"


@dataclass
class ParticleVisual:
    """Visual representation of a particle for rendering.

    Particles flow along links with fading trails.
    """

    id: str
    particle_type: str

    # Position interpolated along link path
    x: float
    y: float

    # Visual properties
    color: str = "#ff6b6b"
    size: float = 6.0
    trail: list[tuple[float, float]] = field(default_factory=list)  # Previous positions


@dataclass
class LabelRegionVisual:
    """Visual representation of a label region for rendering.

    Label regions appear as soft translucent clouds around agents sharing a label.
    """

    name: str

    # Convex hull points defining the region boundary
    hull_points: list[tuple[float, float]] = field(default_factory=list)

    # Visual properties
    fill_color: str = "#88ccff33"  # Semi-transparent
    breathing_phase: float = 0.0


@dataclass
class Frame:
    """A complete visual frame for rendering.

    Contains all visual elements needed to render one animation frame.
    """

    tick: int
    time: float

    agents: list[AgentVisual] = field(default_factory=list)
    links: list[LinkVisual] = field(default_factory=list)
    particles: list[ParticleVisual] = field(default_factory=list)
    label_regions: list[LabelRegionVisual] = field(default_factory=list)


# Particle type to color mapping
PARTICLE_COLORS: dict[str, str] = {
    "order_ticket": "#ff6b6b",  # Red
    "sandwich": "#4ecdc4",  # Teal
    "payment": "#45b7d1",  # Blue
    "directive": "#f9ca24",  # Yellow
    "report": "#a29bfe",  # Purple
    "default": "#888888",  # Gray
}

# Link type to color mapping
LINK_COLORS: dict[str, str] = {
    "hierarchical": "#e74c3c",  # Red
    "peer": "#3498db",  # Blue
    "service": "#2ecc71",  # Green
    "competitive": "#f39c12",  # Orange
}

# Role to color mapping
AGENT_COLORS: dict[str, str] = {
    "owner": "#9b59b6",  # Purple
    "sandwich_maker": "#e67e22",  # Orange
    "cashier": "#27ae60",  # Green
    "customer": "#3498db",  # Blue
    "default": "#4a90d9",  # Default blue
}


def project(world: World) -> Frame:
    """Project World state into a visual Frame.

    Args:
        world: The simulation world state

    Returns:
        Frame containing all visual elements for rendering
    """
    frame = Frame(tick=world.tick, time=world.time)

    # Project agents
    for agent in world.agents.values():
        frame.agents.append(_project_agent(agent, world.tick))

    # Project links
    for link in world.links.values():
        link_visual = _project_link(link, world)
        if link_visual:
            frame.links.append(link_visual)

    # Project particles
    for particle in world.particles.values():
        if particle.alive:
            particle_visual = _project_particle(particle, world)
            if particle_visual:
                frame.particles.append(particle_visual)

    # Project label regions
    frame.label_regions = _project_label_regions(world)

    return frame


def _project_agent(agent: Agent, tick: int) -> AgentVisual:
    """Project an Agent into an AgentVisual.

    Args:
        agent: The agent to project
        tick: Current simulation tick (for breathing phase)

    Returns:
        AgentVisual for rendering
    """
    # Calculate breathing phase based on tick
    breathing_rate = 0.05
    breathing_phase = (tick * breathing_rate) % (2 * math.pi)

    # Calculate glow intensity based on input buffer depth
    # More items in buffer = higher glow
    buffer_depth = len(agent.input_buffer)
    glow_intensity = min(1.0, buffer_depth * 0.2)  # Cap at 1.0

    # Get color based on role
    color = AGENT_COLORS.get(agent.role, AGENT_COLORS["default"])

    return AgentVisual(
        id=agent.id,
        name=agent.name,
        role=agent.role,
        x=agent.x,
        y=agent.y,
        radius=20.0,
        breathing_phase=breathing_phase,
        breathing_rate=breathing_rate,
        color=color,
        glow_intensity=glow_intensity,
        ooda_phase=agent.loop_phase.value,
        labels=list(agent.labels),
    )


def _project_link(link: Link, world: World) -> LinkVisual | None:
    """Project a Link into a LinkVisual.

    Args:
        link: The link to project
        world: The world containing agents

    Returns:
        LinkVisual for rendering, or None if agents not found
    """
    source = world.agents.get(link.source_id)
    dest = world.agents.get(link.dest_id)

    if source is None or dest is None:
        return None

    # Calculate bezier control points
    control_points = _compute_bezier_control_points(source.x, source.y, dest.x, dest.y)

    # Calculate thickness based on recent particles (use property if available)
    thickness = 2.0
    bandwidth = link.properties.get("bandwidth", 1.0)
    thickness = max(1.0, min(6.0, 2.0 * bandwidth))

    # Get color based on link type
    link_type_str = link.link_type.value
    color = LINK_COLORS.get(link_type_str, "#666666")

    # Calculate sway phase based on world tick
    sway_phase = (world.tick * 0.02) % (2 * math.pi)

    return LinkVisual(
        id=link.id,
        source_id=link.source_id,
        dest_id=link.dest_id,
        link_type=link_type_str,
        control_points=control_points,
        thickness=thickness,
        sway_phase=sway_phase,
        color=color,
    )


def _compute_bezier_control_points(
    x1: float, y1: float, x2: float, y2: float
) -> list[tuple[float, float]]:
    """Compute bezier curve control points for a link.

    Creates a smooth curve with two control points offset perpendicular
    to the line between source and dest.

    Args:
        x1, y1: Source position
        x2, y2: Destination position

    Returns:
        List of 4 points: [start, control1, control2, end]
    """
    # Direction vector
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)

    if length < 0.001:
        # Degenerate case: same position
        return [(x1, y1), (x1, y1), (x2, y2), (x2, y2)]

    # Normalized direction
    nx = dx / length
    ny = dy / length

    # Perpendicular vector (for control point offset)
    px = -ny
    py = nx

    # Control point offset (curve amount)
    curve_amount = min(30.0, length * 0.2)

    # Control points at 1/3 and 2/3 along the line, offset perpendicular
    ctrl1_x = x1 + dx * 0.33 + px * curve_amount
    ctrl1_y = y1 + dy * 0.33 + py * curve_amount
    ctrl2_x = x1 + dx * 0.67 + px * curve_amount
    ctrl2_y = y1 + dy * 0.67 + py * curve_amount

    return [
        (x1, y1),
        (ctrl1_x, ctrl1_y),
        (ctrl2_x, ctrl2_y),
        (x2, y2),
    ]


def _project_particle(particle: Particle, world: World) -> ParticleVisual | None:
    """Project a Particle into a ParticleVisual.

    Interpolates particle position along its link path based on progress.

    Args:
        particle: The particle to project
        world: The world containing links and agents

    Returns:
        ParticleVisual for rendering, or None if position cannot be determined
    """
    # Get particle position based on progress along link
    x, y = _interpolate_particle_position(particle, world)

    if x is None or y is None:
        return None

    # Get color based on particle type
    color = PARTICLE_COLORS.get(particle.particle_type, PARTICLE_COLORS["default"])

    return ParticleVisual(
        id=particle.id,
        particle_type=particle.particle_type,
        x=x,
        y=y,
        color=color,
        size=6.0,
        trail=[],  # Trail would be populated by accumulating previous positions
    )


def _interpolate_particle_position(
    particle: Particle, world: World
) -> tuple[float | None, float | None]:
    """Calculate particle position based on progress along link.

    Uses cubic bezier interpolation along the link path.

    Args:
        particle: The particle to position
        world: The world containing links and agents

    Returns:
        (x, y) position, or (None, None) if cannot be determined
    """
    # Get link the particle is traveling on
    link = world.links.get(particle.link_id)

    if link is None:
        # Particle not on a link - try to use source/dest agents
        source = world.agents.get(particle.source_id)
        dest = world.agents.get(particle.dest_id)

        if source is None and dest is None:
            return None, None

        if source and dest:
            # Interpolate directly between source and dest
            progress = particle.progress
            x = source.x + (dest.x - source.x) * progress
            y = source.y + (dest.y - source.y) * progress
            return x, y

        # At source or dest
        if source:
            return source.x, source.y
        if dest:
            return dest.x, dest.y

        return None, None

    # Get source and dest positions
    source = world.agents.get(link.source_id)
    dest = world.agents.get(link.dest_id)

    if source is None or dest is None:
        return None, None

    # Get bezier control points
    control_points = _compute_bezier_control_points(source.x, source.y, dest.x, dest.y)

    # Interpolate along bezier curve
    t = particle.progress
    x, y = _bezier_point(control_points, t)

    return x, y


def _bezier_point(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    """Calculate point on cubic bezier curve at parameter t.

    Args:
        points: List of 4 control points [(x, y), ...]
        t: Parameter from 0 to 1

    Returns:
        (x, y) point on curve
    """
    if len(points) != 4:
        # Fallback for invalid input
        return points[0] if points else (0.0, 0.0)

    # Cubic bezier formula: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    x = (
        mt3 * points[0][0]
        + 3 * mt2 * t * points[1][0]
        + 3 * mt * t2 * points[2][0]
        + t3 * points[3][0]
    )
    y = (
        mt3 * points[0][1]
        + 3 * mt2 * t * points[1][1]
        + 3 * mt * t2 * points[2][1]
        + t3 * points[3][1]
    )

    return x, y


def _project_label_regions(world: World) -> list[LabelRegionVisual]:
    """Project label regions as convex hulls around agents sharing labels.

    Args:
        world: The world containing agents and labels

    Returns:
        List of LabelRegionVisual for rendering
    """
    regions = []

    # Group agents by label
    label_agents: dict[str, list[tuple[float, float]]] = {}

    for agent in world.agents.values():
        for label in agent.labels:
            if label not in label_agents:
                label_agents[label] = []
            label_agents[label].append((agent.x, agent.y))

    # Create region for each label with 2+ agents
    for label, positions in label_agents.items():
        if len(positions) < 2:
            continue

        # Compute convex hull
        hull_points = _convex_hull(positions)

        # Expand hull slightly for visual padding
        hull_points = _expand_hull(hull_points, padding=30.0)

        # Get label-specific color (or default)
        fill_color = "#88ccff33"

        regions.append(
            LabelRegionVisual(
                name=label,
                hull_points=hull_points,
                fill_color=fill_color,
                breathing_phase=(world.tick * 0.03) % (2 * math.pi),
            )
        )

    return regions


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute convex hull of a set of points using Andrew's monotone chain algorithm.

    Args:
        points: List of (x, y) points

    Returns:
        List of points forming the convex hull in counter-clockwise order
    """
    if len(points) <= 2:
        return list(points)

    # Sort points by x, then y
    sorted_points = sorted(points)

    # Build lower hull
    lower: list[tuple[float, float]] = []
    for p in sorted_points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper: list[tuple[float, float]] = []
    for p in reversed(sorted_points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove last point of each half (duplicate of first point of other half)
    return lower[:-1] + upper[:-1]


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate cross product of vectors OA and OB.

    Returns positive if counter-clockwise, negative if clockwise, 0 if collinear.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _expand_hull(hull: list[tuple[float, float]], padding: float) -> list[tuple[float, float]]:
    """Expand convex hull outward by padding amount.

    Args:
        hull: Convex hull points in counter-clockwise order
        padding: Amount to expand

    Returns:
        Expanded hull points
    """
    if len(hull) < 3:
        return hull

    # Calculate centroid
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)

    # Expand each point away from centroid
    expanded = []
    for px, py in hull:
        dx = px - cx
        dy = py - cy
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > 0:
            # Move point outward
            nx = px + (dx / dist) * padding
            ny = py + (dy / dist) * padding
            expanded.append((nx, ny))
        else:
            expanded.append((px, py))

    return expanded


def compute_breathing_radius(base_radius: float, phase: float) -> float:
    """Calculate current radius with breathing animation.

    Args:
        base_radius: Base radius of the agent
        phase: Current breathing phase (0 to 2π)

    Returns:
        Animated radius with 5% oscillation
    """
    return base_radius * (1.0 + 0.05 * math.sin(phase))

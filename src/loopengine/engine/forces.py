"""Force-directed layout engine for agent positioning per PRD section 4.4.

Computes four forces per agent:
1. Link spring force - attraction along links proportional to interaction density
2. Agent repulsion - inverse-square repulsion to prevent overlap
3. Hierarchical vertical bias - upward for authority, downward for subordinate
4. Label cohesion - pull toward centroid of label group

Forces are summed, integrated into velocity (Euler), damped, and applied to position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loopengine.model.world import World

from loopengine.model.link import LinkType


@dataclass
class ForceConfig:
    """Configuration for force-directed layout parameters."""

    # Link spring force
    spring_strength: float = 0.01  # Base spring constant
    spring_rest_length: float = 100.0  # Natural spring length

    # Agent repulsion
    repulsion_strength: float = 1000.0  # Inverse-square repulsion constant
    min_distance: float = 10.0  # Minimum distance to prevent singularities

    # Hierarchical vertical bias
    vertical_strength: float = 0.5  # Vertical force magnitude
    vertical_separation: float = 50.0  # Desired vertical separation

    # Label cohesion
    cohesion_strength: float = 0.005  # Pull toward label centroid

    # Integration parameters
    damping: float = 0.9  # Velocity damping per tick (0-1, closer to 1 = more damping)
    max_velocity: float = 50.0  # Maximum velocity magnitude
    max_force: float = 100.0  # Maximum force magnitude per tick


# Default configuration
DEFAULT_CONFIG = ForceConfig()


@dataclass
class ForceVector:
    """A 2D force vector."""

    fx: float = 0.0
    fy: float = 0.0

    def add(self, other: ForceVector) -> ForceVector:
        """Add another force vector to this one."""
        return ForceVector(self.fx + other.fx, self.fy + other.fy)

    def magnitude(self) -> float:
        """Calculate the magnitude of the force."""
        return math.sqrt(self.fx * self.fx + self.fy * self.fy)

    def clamp(self, max_mag: float) -> ForceVector:
        """Clamp the force magnitude to a maximum value."""
        mag = self.magnitude()
        if mag > max_mag and mag > 0:
            scale = max_mag / mag
            return ForceVector(self.fx * scale, self.fy * scale)
        return self


@dataclass
class AgentForces:
    """Accumulated forces for all agents."""

    forces: dict[str, ForceVector] = field(default_factory=dict)

    def get(self, agent_id: str) -> ForceVector:
        """Get force vector for an agent, creating if necessary."""
        if agent_id not in self.forces:
            self.forces[agent_id] = ForceVector()
        return self.forces[agent_id]

    def add_force(self, agent_id: str, force: ForceVector) -> None:
        """Add a force to an agent's accumulated force."""
        current = self.get(agent_id)
        self.forces[agent_id] = current.add(force)


def compute_forces(world: World, config: ForceConfig | None = None) -> AgentForces:
    """Compute force vectors for all agents in the world.

    Args:
        world: The world containing agents, links, and labels
        config: Force configuration (uses DEFAULT_CONFIG if None)

    Returns:
        AgentForces with force vectors per agent
    """
    if config is None:
        config = DEFAULT_CONFIG

    forces = AgentForces()

    # Initialize forces for all agents
    for agent_id in world.agents:
        forces.get(agent_id)

    # Compute each force type
    _compute_spring_forces(world, forces, config)
    _compute_repulsion_forces(world, forces, config)
    _compute_hierarchical_forces(world, forces, config)
    _compute_cohesion_forces(world, forces, config)

    return forces


def _compute_spring_forces(world: World, forces: AgentForces, config: ForceConfig) -> None:
    """Compute link spring forces pulling connected agents together.

    Spring strength is proportional to interaction density (recent particles).
    """
    for link in world.links.values():
        source = world.agents.get(link.source_id)
        dest = world.agents.get(link.dest_id)

        if source is None or dest is None:
            continue

        # Calculate distance between agents
        dx = dest.x - source.x
        dy = dest.y - source.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < config.min_distance:
            distance = config.min_distance

        # Normalize direction
        nx = dx / distance
        ny = dy / distance

        # Spring force: F = k * (distance - rest_length)
        # Positive when stretched (attraction), negative when compressed (repulsion)
        displacement = distance - config.spring_rest_length
        force_magnitude = config.spring_strength * displacement

        # Apply force to both agents (Newton's third law)
        # Source is pulled toward dest
        forces.add_force(link.source_id, ForceVector(force_magnitude * nx, force_magnitude * ny))
        # Dest is pulled toward source
        forces.add_force(link.dest_id, ForceVector(-force_magnitude * nx, -force_magnitude * ny))


def _compute_repulsion_forces(world: World, forces: AgentForces, config: ForceConfig) -> None:
    """Compute agent repulsion forces to prevent overlap.

    Uses inverse-square law: F = k / d^2
    """
    agents = list(world.agents.values())

    for i, agent_a in enumerate(agents):
        for agent_b in agents[i + 1 :]:
            # Calculate distance between agents
            dx = agent_b.x - agent_a.x
            dy = agent_b.y - agent_a.y
            distance_sq = dx * dx + dy * dy
            distance = math.sqrt(distance_sq)

            if distance < config.min_distance:
                distance = config.min_distance
                distance_sq = distance * distance
                # When too close, use random direction to prevent stuck state
                if dx == 0 and dy == 0:
                    dx = 1.0
                    dy = 0.0

            # Normalize direction
            nx = dx / distance
            ny = dy / distance

            # Inverse-square repulsion: F = k / d^2
            force_magnitude = config.repulsion_strength / distance_sq

            # Apply repulsion in opposite directions
            # Agent A is pushed away from B (opposite to direction toward B)
            forces.add_force(agent_a.id, ForceVector(-force_magnitude * nx, -force_magnitude * ny))
            # Agent B is pushed away from A (toward direction from A)
            forces.add_force(agent_b.id, ForceVector(force_magnitude * nx, force_magnitude * ny))


def _compute_hierarchical_forces(world: World, forces: AgentForces, config: ForceConfig) -> None:
    """Compute hierarchical vertical bias forces.

    For hierarchical links:
    - Authority (source) gets upward force
    - Subordinate (dest) gets downward force
    """
    for link in world.links.values():
        if link.link_type != LinkType.HIERARCHICAL:
            continue

        source = world.agents.get(link.source_id)
        dest = world.agents.get(link.dest_id)

        if source is None or dest is None:
            continue

        # Calculate current vertical separation
        vertical_diff = source.y - dest.y  # Positive if source is above dest

        # We want source (authority) above dest (subordinate)
        # If source.y < dest.y, source needs to move up (negative y)
        # Note: In screen coordinates, smaller y = higher on screen
        # We'll use math coordinates where larger y = higher

        # Apply vertical bias based on desired separation
        # If already separated correctly, reduce force
        target_separation = config.vertical_separation
        separation_error = target_separation - vertical_diff

        # Force proportional to how much separation is needed
        vertical_force = config.vertical_strength * separation_error

        # Push source up (positive y) and dest down (negative y)
        forces.add_force(link.source_id, ForceVector(0.0, vertical_force / 2))
        forces.add_force(link.dest_id, ForceVector(0.0, -vertical_force / 2))


def _compute_cohesion_forces(world: World, forces: AgentForces, config: ForceConfig) -> None:
    """Compute label cohesion forces pulling agents toward label group centroids.

    For each label, compute the centroid of all agents with that label,
    then apply a force pulling each agent toward that centroid.
    """
    # Group agents by label
    label_agents: dict[str, list[str]] = {}

    for agent in world.agents.values():
        for label in agent.labels:
            if label not in label_agents:
                label_agents[label] = []
            label_agents[label].append(agent.id)

    # For each label, compute centroid and apply cohesion
    for _label, agent_ids in label_agents.items():
        if len(agent_ids) < 2:
            continue  # No cohesion needed for single agent

        # Compute centroid
        centroid_x = 0.0
        centroid_y = 0.0

        for agent_id in agent_ids:
            agent = world.agents[agent_id]
            centroid_x += agent.x
            centroid_y += agent.y

        centroid_x /= len(agent_ids)
        centroid_y /= len(agent_ids)

        # Apply cohesion force to each agent
        for agent_id in agent_ids:
            agent = world.agents[agent_id]

            dx = centroid_x - agent.x
            dy = centroid_y - agent.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < config.min_distance:
                continue  # Already at centroid

            # Linear pull toward centroid
            force_magnitude = config.cohesion_strength * distance

            # Normalize and apply
            nx = dx / distance
            ny = dy / distance

            forces.add_force(agent_id, ForceVector(force_magnitude * nx, force_magnitude * ny))


def apply_forces(world: World, forces: AgentForces, config: ForceConfig | None = None) -> None:
    """Apply computed forces to agent velocities and positions.

    Uses Euler integration with damping:
    1. Clamp forces to max_force
    2. Add force to velocity
    3. Apply damping to velocity
    4. Clamp velocity to max_velocity
    5. Add velocity to position

    Args:
        world: The world containing agents
        forces: Computed forces per agent
        config: Force configuration
    """
    if config is None:
        config = DEFAULT_CONFIG

    for agent_id, force in forces.forces.items():
        agent = world.agents.get(agent_id)
        if agent is None:
            continue

        # Clamp force
        clamped_force = force.clamp(config.max_force)

        # Update velocity: v = v + F (assuming unit mass)
        agent.vx += clamped_force.fx
        agent.vy += clamped_force.fy

        # Apply damping: v = v * (1 - damping)
        damping_factor = 1.0 - config.damping
        agent.vx *= damping_factor
        agent.vy *= damping_factor

        # Clamp velocity
        velocity_mag = math.sqrt(agent.vx * agent.vx + agent.vy * agent.vy)
        if velocity_mag > config.max_velocity:
            scale = config.max_velocity / velocity_mag
            agent.vx *= scale
            agent.vy *= scale

        # Update position
        agent.x += agent.vx
        agent.y += agent.vy


def update_layout(world: World, config: ForceConfig | None = None) -> None:
    """Compute forces and update agent positions in one step.

    This is the main entry point called each tick (or every N ticks).

    Args:
        world: The world to update
        config: Force configuration
    """
    if config is None:
        config = DEFAULT_CONFIG

    forces = compute_forces(world, config)
    apply_forces(world, forces, config)

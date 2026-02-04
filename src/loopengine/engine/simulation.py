"""World simulation tick driver: executes one simulation tick per PRD section 4.2."""

from __future__ import annotations

import logging
import random
import uuid
from typing import TYPE_CHECKING

from loopengine.engine.forces import update_layout
from loopengine.engine.loop import step_agent
from loopengine.model.particle import Particle

if TYPE_CHECKING:
    from loopengine.model.world import ExternalInput, World

logger = logging.getLogger(__name__)

# Performance optimization constants
MAX_PARTICLES = 100  # Maximum concurrent particles (excess queued)
FORCE_UPDATE_INTERVAL = 1  # Update forces every N ticks (1 = every tick)


def tick_world(world: World) -> None:
    """Execute one simulation tick per PRD tick sequence.

    Tick sequence:
    1. Generate external inputs (roll against rate * schedule)
    2. Step all agents, collect emitted particles
    3. Advance all particles (progress += speed)
    4. Deliver particles where progress >= 1.0
    5. Garbage collect dead particles
    6. Update force-directed layout (compute forces, update positions)
    7. Increment world.tick and world.time

    Args:
        world: The world to advance

    Side effects:
        - Mutates world.tick and world.time
        - Mutates world.particles
        - Mutates agent input_buffer and output_buffer
        - May add particles from external inputs
    """
    # 1. Generate external inputs
    _generate_external_inputs(world)

    # 2. Step all agents, collect emitted particles
    _step_all_agents(world)

    # 3. Advance all particles along links
    _advance_particles(world)

    # 4. Deliver particles where progress >= 1.0
    _deliver_particles(world)

    # 5. Garbage collect dead particles
    _garbage_collect_particles(world)

    # 6. Update force layout (stub for now)
    _update_force_layout(world)

    # 7. Increment world.tick and world.time
    world.tick += 1
    world.time += 1.0 / world.speed

    # Debug log every 100 ticks to avoid log spam
    if world.tick % 100 == 0:
        logger.debug(
            "Simulation tick %d: agents=%d, particles=%d",
            world.tick,
            len(world.agents),
            len(world.particles),
        )


def _generate_external_inputs(world: World) -> None:
    """Generate particles from external inputs based on rate and schedule.

    For each external input, roll against rate * schedule(tick) * (1 + variance * random).
    If roll succeeds, create a particle targeting the specified agent.
    """
    for ext_input in world.external_inputs:
        _maybe_spawn_particle(world, ext_input)


def _maybe_spawn_particle(world: World, ext_input: ExternalInput) -> None:
    """Possibly spawn a particle from an external input.

    Uses Poisson-like probability based on rate, schedule, and variance.
    """
    # Calculate effective rate at this tick
    schedule_mult = ext_input.schedule(world.tick)
    variance_mult = 1.0 + ext_input.variance * (random.random() * 2 - 1)
    effective_rate = ext_input.rate * schedule_mult * variance_mult

    # Roll against rate (probability per tick)
    if random.random() < effective_rate:
        # Spawn particle
        particle = Particle(
            id=str(uuid.uuid4()),
            particle_type=ext_input.particle_type,
            payload=ext_input.payload_generator(),
            source_id="external",
            dest_id=ext_input.target_agent_id,
            link_id="",  # External inputs bypass links, go directly to input buffer
            progress=1.0,  # Ready for immediate delivery
            speed=0.0,
            alive=True,
        )

        # Deliver directly to target agent's input buffer
        if ext_input.target_agent_id in world.agents:
            world.agents[ext_input.target_agent_id].input_buffer.append(particle)


def _step_all_agents(world: World) -> None:
    """Step all agents through their OODA loops, collect emitted particles."""
    for agent in world.agents.values():
        # Step the agent
        new_particles = step_agent(agent, world)

        # Place new particles into world.particles and assign to links
        for particle in new_particles:
            _place_particle_on_link(world, particle)

        # Also move particles from output buffer to world.particles
        for particle in list(agent.output_buffer):
            if particle.id not in world.particles:
                _place_particle_on_link(world, particle)
        agent.output_buffer.clear()


def _place_particle_on_link(world: World, particle: Particle) -> None:
    """Place a particle on the appropriate link based on source and destination.

    Enforces MAX_PARTICLES limit. When limit is reached, new particles are dropped
    to maintain performance. The oldest particles (by progress) are kept.
    """
    # Enforce particle count limit - drop new particles if at capacity
    if len(world.particles) >= MAX_PARTICLES:
        # Log at debug level to avoid spam
        logger.debug(
            "Particle limit (%d) reached, dropping particle %s",
            MAX_PARTICLES,
            particle.id,
        )
        particle.alive = False
        return

    # Find a link from source to destination
    link_id = None
    for link in world.links.values():
        if link.source_id == particle.source_id and link.dest_id == particle.dest_id:
            link_id = link.id
            break

    if link_id:
        particle.link_id = link_id
        particle.progress = 0.0  # Start at beginning of link
        world.particles[particle.id] = particle
    else:
        # No direct link found - particle cannot be routed
        # Mark as dead
        particle.alive = False


def _advance_particles(world: World) -> None:
    """Advance all active particles along their links."""
    for particle in world.particles.values():
        if particle.alive and particle.progress < 1.0:
            particle.progress += particle.speed


def _deliver_particles(world: World) -> None:
    """Deliver particles that have reached their destination (progress >= 1.0)."""
    for particle in world.particles.values():
        if particle.alive and particle.progress >= 1.0:
            # Deliver to destination agent
            if particle.dest_id in world.agents:
                world.agents[particle.dest_id].input_buffer.append(particle)
            # Mark as delivered (dead)
            particle.alive = False


def _garbage_collect_particles(world: World) -> None:
    """Remove dead particles from world.particles."""
    dead_ids = [pid for pid, p in world.particles.items() if not p.alive]
    for pid in dead_ids:
        del world.particles[pid]


def _update_force_layout(world: World) -> None:
    """Update force-directed layout for agent positions.

    Computes forces on all agents and updates their positions each tick.
    """
    update_layout(world)


def tick_world_with_flexibility(
    world: World,
    flexibility_by_role: dict[str, float] | None = None,
) -> bool:
    """Execute one simulation tick with flexibility-based perturbations.

    This enhanced tick function injects random perturbations based on the
    flexibility scores of roles in the simulation. Higher flexibility roles
    will experience more perturbations/surprises.

    Args:
        world: The world to advance.
        flexibility_by_role: Optional dict mapping role names to flexibility scores.
                           If None, uses flexibility from world.schemas.

    Returns:
        True if any perturbation was injected this tick.

    Side effects:
        Same as tick_world, plus potential perturbation injections.
    """
    from loopengine.engine.flexibility import inject_perturbation

    # Calculate average flexibility for perturbation probability
    if flexibility_by_role is None:
        flexibility_by_role = {
            role: schema.flexibility_score for role, schema in world.schemas.items()
        }

    # Use max flexibility across roles for perturbation chance
    # This ensures high-flexibility roles get their expected surprises
    if flexibility_by_role:
        max_flexibility = max(flexibility_by_role.values())
    else:
        max_flexibility = 0.5  # Default if no schemas

    # Inject perturbation based on flexibility
    perturbation_injected = inject_perturbation(world, max_flexibility)

    # Run normal tick
    tick_world(world)

    return perturbation_injected

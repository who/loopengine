"""Agent OODA loop stepper: advances agents through Sense→Orient→Decide→Act phases."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from loopengine.model.agent import Phase

if TYPE_CHECKING:
    from loopengine.model.agent import Agent
    from loopengine.model.particle import Particle
    from loopengine.model.world import World

logger = logging.getLogger(__name__)


def step_agent(agent: Agent, world: World) -> list[Particle]:
    """Advance an agent one step in its OODA loop.

    This is the innermost loop of the simulation. Each call advances the agent
    through one tick of its current phase.

    Args:
        agent: The agent to step
        world: The world context (used for link lookup, etc.)

    Returns:
        List of newly created particles (only populated during ACT phase)

    Side effects:
        - Mutates agent.internal_state
        - Mutates agent.input_buffer (clears during SENSE)
        - Mutates agent.loop_phase
        - Mutates agent.phase_tick
    """
    new_particles: list[Particle] = []

    # Execute current phase behavior
    if agent.loop_phase == Phase.SENSE:
        _do_sense(agent)
    elif agent.loop_phase == Phase.ORIENT:
        _do_orient(agent)
    elif agent.loop_phase == Phase.DECIDE:
        _do_decide(agent)
    elif agent.loop_phase == Phase.ACT:
        new_particles = _do_act(agent, world)

    # Advance phase timing
    _advance_phase(agent)

    return new_particles


def _do_sense(agent: Agent) -> None:
    """SENSE phase: Read input_buffer into internal_state, clear buffer.

    Only executes on the first tick of the phase (phase_tick == 0) to prevent
    overwriting sensed_inputs with an empty buffer on subsequent ticks.

    Genome traits like 'observation' and 'signal_discrimination' could filter
    or weight the sensed inputs (not implemented in base version).
    """
    if agent.phase_tick != 0:
        return  # Only sense at start of phase
    # Store sensed inputs in internal state
    agent.internal_state["sensed_inputs"] = list(agent.input_buffer)
    # Clear the input buffer
    agent.input_buffer.clear()


def _do_orient(agent: Agent) -> None:
    """ORIENT phase: Interpret sensed_inputs through genome biases.

    Only executes on the first tick of the phase (phase_tick == 0).

    Risk-tolerant agents might interpret ambiguous signals as opportunities.
    Update internal_state with oriented interpretation.
    """
    if agent.phase_tick != 0:
        return  # Only orient at start of phase
    # Base implementation: just pass through sensed inputs as oriented_inputs
    # Genome biases would modify interpretation here in advanced implementations
    sensed = agent.internal_state.get("sensed_inputs", [])
    agent.internal_state["oriented_inputs"] = sensed


def _do_decide(agent: Agent) -> None:
    """DECIDE phase: Run policy to produce planned_actions.

    Only executes on the first tick of the phase (phase_tick == 0) to prevent
    running the policy multiple times per cycle.

    Invokes agent.policy(sensed_inputs, genome, internal_state) → planned_actions.
    Stores planned_actions in internal_state.

    If the policy raises an exception, the error is logged and the agent continues
    with an empty action list. This ensures simulation continuity even when
    individual agent policies fail.
    """
    if agent.phase_tick != 0:
        return  # Only decide at start of phase
    if agent.policy is None:
        agent.internal_state["planned_actions"] = []
        return

    sensed_inputs = agent.internal_state.get("sensed_inputs", [])
    try:
        planned_actions = agent.policy(sensed_inputs, agent.genome, agent.internal_state)
        agent.internal_state["planned_actions"] = planned_actions
    except Exception:
        # Log error with agent context, continue with empty actions
        logger.exception(
            "Policy execution failed for agent '%s' (role=%s). Continuing with empty action list.",
            agent.id,
            agent.role,
        )
        agent.internal_state["planned_actions"] = []
        # Track policy failure count for debugging
        agent.internal_state["_policy_failures"] = (
            agent.internal_state.get("_policy_failures", 0) + 1
        )


def _do_act(agent: Agent, world: World) -> list[Particle]:
    """ACT phase: Convert planned_actions into Particles.

    Only executes on the first tick of the phase (phase_tick == 0) to prevent
    emitting duplicate particles on subsequent ticks.

    Places particles on appropriate outgoing links. Returns newly created particles.
    """
    if agent.phase_tick != 0:
        return []  # Only act at start of phase
    planned_actions = agent.internal_state.get("planned_actions", [])

    # The planned_actions should already be Particle objects from the policy
    # Copy them to new_particles list and place in agent's output buffer
    new_particles: list[Particle] = []
    for action in planned_actions:
        # Policy returns Particle objects
        new_particles.append(action)
        agent.output_buffer.append(action)

    # Clear planned actions after acting
    agent.internal_state["planned_actions"] = []

    return new_particles


def _advance_phase(agent: Agent) -> None:
    """Advance phase timing. Move to next phase when current phase completes."""
    agent.phase_tick += 1

    # Calculate phase duration (ticks per phase)
    phase_duration = agent.loop_period // 4

    if agent.phase_tick >= phase_duration:
        # Time to advance to next phase
        agent.phase_tick = 0
        agent.loop_phase = _next_phase(agent.loop_phase)


def _next_phase(phase: Phase) -> Phase:
    """Get the next phase in the OODA cycle."""
    phase_order = [Phase.SENSE, Phase.ORIENT, Phase.DECIDE, Phase.ACT]
    current_index = phase_order.index(phase)
    next_index = (current_index + 1) % len(phase_order)
    return phase_order[next_index]

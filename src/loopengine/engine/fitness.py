"""Role-specific fitness functions for genetic algorithm evaluation.

Fitness functions compute a scalar score representing how well an agent
performed its role during simulation. These are used by the GA engine
for selection.

Each fitness function follows the signature:
    fitness_fn(world: World, agent_id: str) -> float
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loopengine.model.world import World


def tom_fitness(world: World, agent_id: str) -> float:
    """Compute fitness for Tom (sandwich_maker role) per PRD section 9.6.

    Fitness formula:
        fitness = (sandwiches_completed / ticks_elapsed)  # throughput
                * average_consistency_score                # quality
                - (waste_particles / total_ingredients_used)  # efficiency
                - (max_queue_depth * 0.1)                  # responsiveness penalty

    The function reads metrics from agent internal_state that are tracked
    during simulation by tom_policy and alex_policy.

    Args:
        world: World state after simulation.
        agent_id: ID of the agent to evaluate (should be Tom).

    Returns:
        float: Scalar fitness value (higher is better).

    Raises:
        ValueError: If agent not found in world.
    """
    if agent_id not in world.agents:
        msg = f"Agent '{agent_id}' not found in world"
        raise ValueError(msg)

    agent = world.agents[agent_id]
    internal_state = agent.internal_state
    ticks_elapsed = world.tick

    # Avoid division by zero
    if ticks_elapsed == 0:
        return 0.0

    # Get metrics from Tom's internal_state (tracked by tom_policy)
    sandwiches_completed = internal_state.get("sandwiches_completed", 0)
    quality_scores = internal_state.get("quality_scores", [])
    waste_count = internal_state.get("waste_count", 0)
    ingredients_used = internal_state.get("ingredients_used", 0)

    # Get max_queue_depth from Alex's internal_state (tracked by alex_policy)
    alex = world.agents.get("alex")
    max_queue_depth = 0
    if alex:
        max_queue_depth = alex.internal_state.get("max_queue_depth", 0)

    # Compute throughput: sandwiches per tick
    throughput = sandwiches_completed / ticks_elapsed

    # Compute average consistency score (quality)
    if quality_scores:
        avg_consistency = sum(quality_scores) / len(quality_scores)
    else:
        avg_consistency = 0.0

    # Compute waste efficiency: waste / total ingredients
    if ingredients_used > 0:
        waste_ratio = waste_count / ingredients_used
    else:
        waste_ratio = 0.0

    # Compute fitness per PRD formula
    fitness = (throughput * avg_consistency) - waste_ratio - (max_queue_depth * 0.1)

    return fitness

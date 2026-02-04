"""Flexibility-based input randomization for GA evaluation and live simulation.

This module implements the flexibility parameter behavior per PRD section 5.3:
- Low flexibility (0.0-0.3): Predictable, consistent inputs across evaluations
- Medium flexibility (0.4-0.6): Some variance but within expected patterns
- High flexibility (0.7-1.0): Varied inputs, selecting for robustness

During GA evaluation:
- Low flexibility roles get the same input stream each evaluation (fixed seed)
- High flexibility roles get varied inputs across evaluations

During live simulation:
- Inject occasional surprises/perturbations proportional to flexibility
"""

from __future__ import annotations

import copy
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loopengine.model.world import World

logger = logging.getLogger(__name__)


@dataclass
class FlexibilityConfig:
    """Configuration for flexibility-based input randomization.

    Attributes:
        base_seed: Base seed for deterministic evaluation. None for random.
        evaluation_runs: Number of runs for high-flexibility evaluation averaging.
        perturbation_base_rate: Base probability of perturbation per tick.
        variance_multiplier_low: Variance multiplier for low flexibility roles.
        variance_multiplier_high: Variance multiplier for high flexibility roles.
    """

    base_seed: int | None = 42
    evaluation_runs: int = 3
    perturbation_base_rate: float = 0.01
    variance_multiplier_low: float = 0.1
    variance_multiplier_high: float = 2.0


def compute_effective_seed(
    base_seed: int | None,
    flexibility: float,
    evaluation_index: int = 0,
) -> int | None:
    """Compute effective seed based on flexibility score.

    For low flexibility (< 0.4): Always use base_seed for consistent inputs.
    For medium flexibility (0.4-0.6): Use base_seed + evaluation_index (some variance).
    For high flexibility (> 0.6): No seed (fully random) or varied seeds.

    Args:
        base_seed: Base seed for deterministic evaluation.
        flexibility: Flexibility score 0.0-1.0.
        evaluation_index: Index for multi-run averaging (0 for single run).

    Returns:
        Computed seed or None for random behavior.
    """
    if base_seed is None:
        return None

    if flexibility < 0.4:
        # Low flexibility: consistent inputs (same seed every time)
        return base_seed
    elif flexibility <= 0.6:
        # Medium flexibility: some variance based on evaluation index
        return base_seed + evaluation_index
    else:
        # High flexibility: varied inputs (different seed each run, or None)
        if evaluation_index == 0:
            return base_seed + random.randint(0, 10000)
        return None


def compute_variance_multiplier(flexibility: float, config: FlexibilityConfig) -> float:
    """Compute variance multiplier for external inputs based on flexibility.

    Low flexibility: Reduce variance (more predictable inputs).
    High flexibility: Increase variance (more unpredictable inputs).

    Args:
        flexibility: Flexibility score 0.0-1.0.
        config: Flexibility configuration.

    Returns:
        Multiplier to apply to external input variance.
    """
    # Linear interpolation from low to high multiplier based on flexibility
    return (
        config.variance_multiplier_low
        + (config.variance_multiplier_high - config.variance_multiplier_low) * flexibility
    )


def adjust_external_input_variance(
    world: World,
    flexibility: float,
    config: FlexibilityConfig | None = None,
) -> None:
    """Adjust external input variance in world based on flexibility score.

    Modifies external_inputs in place to scale variance by flexibility.

    Args:
        world: World to modify (mutates in place).
        flexibility: Flexibility score 0.0-1.0.
        config: Optional flexibility configuration.
    """
    if config is None:
        config = FlexibilityConfig()

    multiplier = compute_variance_multiplier(flexibility, config)

    for ext_input in world.external_inputs:
        # Scale the variance - store original if not already stored
        if not hasattr(ext_input, "_original_variance"):
            ext_input._original_variance = ext_input.variance  # type: ignore[attr-defined]
        ext_input.variance = ext_input._original_variance * multiplier  # type: ignore[attr-defined]

    logger.debug(
        "Adjusted external input variance with multiplier %.2f for flexibility %.2f",
        multiplier,
        flexibility,
    )


def inject_perturbation(
    world: World,
    flexibility: float,
    config: FlexibilityConfig | None = None,
) -> bool:
    """Potentially inject a random perturbation during live simulation.

    Higher flexibility = higher chance of perturbation.
    Perturbations can be:
    - Spike in external input rate
    - Temporary modification of constraints
    - Unexpected particle injection

    Args:
        world: World to potentially perturb.
        flexibility: Flexibility score 0.0-1.0.
        config: Optional flexibility configuration.

    Returns:
        True if a perturbation was injected.
    """
    if config is None:
        config = FlexibilityConfig()

    # Probability of perturbation scales with flexibility
    # Low flexibility (0.0): Almost no perturbations
    # High flexibility (1.0): Full perturbation rate
    perturbation_prob = config.perturbation_base_rate * flexibility

    if random.random() >= perturbation_prob:
        return False

    # Choose perturbation type
    perturbation_type = random.choice(["rate_spike", "extra_input"])

    if perturbation_type == "rate_spike" and world.external_inputs:
        # Temporarily spike external input rate
        ext_input = random.choice(world.external_inputs)
        original_rate = ext_input.rate
        ext_input.rate *= 2.0  # Double the rate for this tick
        logger.debug(
            "Injected rate spike perturbation: %s rate %.2f -> %.2f",
            ext_input.name,
            original_rate,
            ext_input.rate,
        )
        # Note: Rate will reset naturally as we don't persist this change

    elif perturbation_type == "extra_input" and world.external_inputs:
        # Force an extra input this tick
        import uuid

        from loopengine.model.particle import Particle

        ext_input = random.choice(world.external_inputs)
        particle = Particle(
            id=str(uuid.uuid4()),
            particle_type=ext_input.particle_type,
            payload=ext_input.payload_generator(),
            source_id="external_perturbation",
            dest_id=ext_input.target_agent_id,
            link_id="",
            progress=1.0,  # Ready for immediate delivery
            speed=0.0,
            alive=True,
        )

        if ext_input.target_agent_id in world.agents:
            world.agents[ext_input.target_agent_id].input_buffer.append(particle)
            logger.debug(
                "Injected extra input perturbation: %s to agent %s",
                ext_input.particle_type,
                ext_input.target_agent_id,
            )

    return True


@dataclass
class FlexibilityAwareEvaluator:
    """Evaluates genome fitness with flexibility-based input variance.

    For low flexibility roles: Uses consistent inputs (fixed seed) for fair comparison.
    For high flexibility roles: Uses varied inputs to select for robustness.
    """

    # Target configuration
    target_agent_id: str = ""
    target_role: str = ""

    # Simulation parameters
    ticks: int = 1000
    base_seed: int | None = 42

    # Fitness computation
    fitness_fn: Callable[[World, str], float] | None = None

    # Flexibility configuration
    flexibility: float = 0.5
    config: FlexibilityConfig = field(default_factory=FlexibilityConfig)

    def evaluate(
        self,
        genome: dict[str, float],
        world_template: World,
    ) -> float:
        """Evaluate fitness with flexibility-aware input variance.

        For low flexibility: Single evaluation with fixed seed.
        For high flexibility: Average multiple evaluations with varied inputs.

        Args:
            genome: Candidate genome to evaluate.
            world_template: World template to clone for simulation.

        Returns:
            Scalar fitness value (higher is better).
        """
        from loopengine.engine.simulation import tick_world

        if not self.target_agent_id:
            msg = "target_agent_id must be set"
            raise ValueError(msg)

        if self.fitness_fn is None:
            msg = "fitness_fn must be set"
            raise ValueError(msg)

        # Determine number of evaluation runs based on flexibility
        # Low flexibility: 1 run (deterministic)
        # High flexibility: multiple runs (average for robustness)
        num_runs = 1 if self.flexibility < 0.6 else self.config.evaluation_runs

        fitness_values: list[float] = []

        for run_idx in range(num_runs):
            # Compute seed for this evaluation
            seed = compute_effective_seed(
                self.base_seed or self.config.base_seed,
                self.flexibility,
                run_idx,
            )

            # Clone world for isolated evaluation
            world = copy.deepcopy(world_template)

            if self.target_agent_id not in world.agents:
                msg = f"Target agent '{self.target_agent_id}' not found in world"
                raise ValueError(msg)

            # Apply candidate genome
            world.agents[self.target_agent_id].genome = genome.copy()

            # Adjust external input variance based on flexibility
            adjust_external_input_variance(world, self.flexibility, self.config)

            # Set random seed
            if seed is not None:
                random.seed(seed)

            # Run simulation
            for _ in range(self.ticks):
                tick_world(world)

            # Compute fitness
            fitness = self.fitness_fn(world, self.target_agent_id)
            fitness_values.append(fitness)

        # Return average fitness for robustness evaluation
        avg_fitness = sum(fitness_values) / len(fitness_values)

        if num_runs > 1:
            logger.debug(
                "Flexibility-aware evaluation: %d runs, fitness values %s, avg %.4f",
                num_runs,
                fitness_values,
                avg_fitness,
            )

        return avg_fitness


def evaluate_with_flexibility(
    genome: dict[str, float],
    world_template: World,
    target_agent_id: str,
    flexibility: float = 0.5,
    ticks: int = 1000,
    fitness_fn: Callable[[World, str], float] | None = None,
    base_seed: int | None = 42,
    config: FlexibilityConfig | None = None,
) -> float:
    """Evaluate fitness with flexibility-based input variance.

    Convenience function that creates a FlexibilityAwareEvaluator.

    Args:
        genome: Candidate genome to evaluate.
        world_template: World template to clone for simulation.
        target_agent_id: ID of the agent to apply genome to.
        flexibility: Flexibility score 0.0-1.0. Lower = consistent inputs.
        ticks: Number of simulation ticks to run.
        fitness_fn: Function(world, agent_id) -> float for computing fitness.
        base_seed: Base random seed for evaluation.
        config: Optional flexibility configuration.

    Returns:
        Scalar fitness value (higher is better).

    Example:
        >>> # Low flexibility role (consistent evaluation)
        >>> fitness = evaluate_with_flexibility(
        ...     genome, world, "tom", flexibility=0.3,
        ...     fitness_fn=tom_fitness
        ... )
        >>> # High flexibility role (varied inputs, robustness selection)
        >>> fitness = evaluate_with_flexibility(
        ...     genome, world, "maria", flexibility=0.8,
        ...     fitness_fn=maria_fitness
        ... )
    """
    if fitness_fn is None:
        # Default fitness: total customers served
        def fitness_fn(world: World, agent_id: str) -> float:
            total_served = sum(
                a.internal_state.get("served_count", 0) for a in world.agents.values()
            )
            return float(total_served)

    evaluator = FlexibilityAwareEvaluator(
        target_agent_id=target_agent_id,
        ticks=ticks,
        base_seed=base_seed,
        fitness_fn=fitness_fn,
        flexibility=flexibility,
        config=config or FlexibilityConfig(),
    )

    return evaluator.evaluate(genome, world_template)


def get_role_flexibility(world: World, role: str) -> float:
    """Get flexibility score for a role from world schemas.

    Args:
        world: World containing schemas.
        role: Role name to look up.

    Returns:
        Flexibility score 0.0-1.0, defaults to 0.5 if not found.
    """
    if role in world.schemas:
        return world.schemas[role].flexibility_score
    return 0.5  # Default medium flexibility

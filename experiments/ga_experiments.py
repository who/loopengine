#!/usr/bin/env python3
"""GA Experiments for Sandwich Shop Roles.

This script conducts GA experiments evolving each role (Tom, Alex, Maria)
for 100 generations, documenting findings with fitness curves.

Per PRD section 10 Phase 3.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from loopengine.corpora.sandwich_shop import create_world
from loopengine.engine.fitness import alex_fitness, maria_fitness, tom_fitness
from loopengine.engine.ga import GAEngine, evaluate_fitness
from loopengine.model.genome import GenomeSchema, GenomeTrait

if TYPE_CHECKING:
    from collections.abc import Callable

    from loopengine.model.world import World

# Experiment configuration
GENERATIONS = 100
POPULATION_SIZE = 50
SELECTION_COUNT = 10
MUTATION_RATE = 0.1
MUTATION_MAGNITUDE = 0.05
TICKS_PER_EVALUATION = 1000
SEED = 42  # For reproducibility


def create_tom_schema() -> GenomeSchema:
    """Create genome schema for Tom (sandwich_maker)."""
    return GenomeSchema(
        role="sandwich_maker",
        traits={
            "speed": GenomeTrait(
                name="speed",
                description="How fast Tom works",
                min_val=0.0,
                max_val=1.0,
                category="physical",
            ),
            "consistency": GenomeTrait(
                name="consistency",
                description="Quality consistency of sandwiches",
                min_val=0.0,
                max_val=1.0,
                category="skill",
            ),
            "ingredient_intuition": GenomeTrait(
                name="ingredient_intuition",
                description="Ability to substitute ingredients creatively",
                min_val=0.0,
                max_val=1.0,
                category="cognitive",
            ),
            "stress_tolerance": GenomeTrait(
                name="stress_tolerance",
                description="Performance under pressure",
                min_val=0.0,
                max_val=1.0,
                category="temperamental",
            ),
            "waste_minimization": GenomeTrait(
                name="waste_minimization",
                description="Material efficiency",
                min_val=0.0,
                max_val=1.0,
                category="skill",
            ),
        },
    )


def create_alex_schema() -> GenomeSchema:
    """Create genome schema for Alex (cashier)."""
    return GenomeSchema(
        role="cashier",
        traits={
            "speed": GenomeTrait(
                name="speed",
                description="How fast Alex processes orders",
                min_val=0.0,
                max_val=1.0,
                category="physical",
            ),
            "accuracy": GenomeTrait(
                name="accuracy",
                description="Order accuracy rate",
                min_val=0.0,
                max_val=1.0,
                category="skill",
            ),
            "friendliness": GenomeTrait(
                name="friendliness",
                description="Customer service demeanor",
                min_val=0.0,
                max_val=1.0,
                category="social",
            ),
            "stress_tolerance": GenomeTrait(
                name="stress_tolerance",
                description="Performance under pressure",
                min_val=0.0,
                max_val=1.0,
                category="temperamental",
            ),
            "upselling": GenomeTrait(
                name="upselling",
                description="Ability to increase order value",
                min_val=0.0,
                max_val=1.0,
                category="skill",
            ),
        },
    )


def create_maria_schema() -> GenomeSchema:
    """Create genome schema for Maria (owner)."""
    return GenomeSchema(
        role="owner",
        traits={
            "supply_forecasting": GenomeTrait(
                name="supply_forecasting",
                description="Ability to predict supply needs",
                min_val=0.0,
                max_val=1.0,
                category="cognitive",
            ),
            "observation": GenomeTrait(
                name="observation",
                description="Awareness of shop state",
                min_val=0.0,
                max_val=1.0,
                category="cognitive",
            ),
            "decisiveness": GenomeTrait(
                name="decisiveness",
                description="Speed of decision-making",
                min_val=0.0,
                max_val=1.0,
                category="temperamental",
            ),
            "delegation": GenomeTrait(
                name="delegation",
                description="Ability to delegate tasks effectively",
                min_val=0.0,
                max_val=1.0,
                category="social",
            ),
            "cost_sensitivity": GenomeTrait(
                name="cost_sensitivity",
                description="Awareness of costs and margins",
                min_val=0.0,
                max_val=1.0,
                category="cognitive",
            ),
        },
    )


def create_fitness_wrapper(
    world_template: World,
    agent_id: str,
    fitness_fn: Callable[[World, str], float],
) -> Callable[[dict[str, float]], float]:
    """Create a fitness wrapper function for the GA engine.

    Args:
        world_template: World template to clone for each evaluation.
        agent_id: ID of the agent to evaluate.
        fitness_fn: Role-specific fitness function.

    Returns:
        Callable that takes a genome and returns fitness.
    """

    def wrapper(genome: dict[str, float]) -> float:
        return evaluate_fitness(
            genome=genome,
            world_template=world_template,
            target_agent_id=agent_id,
            ticks=TICKS_PER_EVALUATION,
            fitness_fn=fitness_fn,
            seed=None,  # Stochastic evaluation
        )

    return wrapper


def run_evolution(
    agent_id: str,
    role: str,
    schema: GenomeSchema,
    fitness_fn: Callable[[World, str], float],
) -> dict:
    """Run GA evolution for a specific role.

    Args:
        agent_id: ID of the agent to evolve.
        role: Role name for display.
        schema: Genome schema for the role.
        fitness_fn: Role-specific fitness function.

    Returns:
        dict with results including best_genome, best_fitness, stats_history.
    """
    print(f"\n{'=' * 60}")
    print(f"Evolving {role.upper()} ({agent_id})")
    print(f"{'=' * 60}")

    # Create fresh world template
    world_template = create_world()

    # Create GA engine
    ga = GAEngine(
        population_size=POPULATION_SIZE,
        selection_count=SELECTION_COUNT,
        mutation_rate=MUTATION_RATE,
        mutation_magnitude=MUTATION_MAGNITUDE,
        selection_type="rank",
        crossover_type="uniform",
    )

    # Initialize population
    ga.initialize_population(schema)

    # Create fitness wrapper
    fitness_wrapper = create_fitness_wrapper(world_template, agent_id, fitness_fn)

    # Run evolution
    print(f"Running {GENERATIONS} generations with population size {POPULATION_SIZE}...")

    result = ga.run(generations=GENERATIONS, fitness_fn=fitness_wrapper)

    # Print final results
    print("\nFinal Results:")
    print(f"  Best Fitness: {result['best_fitness']:.6f}")
    print("  Best Genome:")
    for trait, value in result["best_genome"].items():
        print(f"    {trait}: {value:.4f}")

    # Add stats history as dicts for JSON serialization
    result["stats_history_dicts"] = [asdict(s) for s in result["stats_history"]]

    return result


def save_results(results: dict, output_dir: Path) -> None:
    """Save experiment results to JSON files.

    Args:
        results: Dictionary of results by role.
        output_dir: Directory to save results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for role, result in results.items():
        # Save detailed results
        filename = output_dir / f"{role}_evolution.json"
        data = {
            "role": role,
            "best_genome": result["best_genome"],
            "best_fitness": result["best_fitness"],
            "generations_run": result["generations_run"],
            "config": {
                "population_size": POPULATION_SIZE,
                "selection_count": SELECTION_COUNT,
                "mutation_rate": MUTATION_RATE,
                "mutation_magnitude": MUTATION_MAGNITUDE,
                "ticks_per_evaluation": TICKS_PER_EVALUATION,
            },
            "stats_history": result["stats_history_dicts"],
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {role} results to {filename}")


def print_summary(results: dict) -> None:
    """Print experiment summary comparing all roles.

    Args:
        results: Dictionary of results by role.
    """
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")

    for role, result in results.items():
        print(f"\n{role.upper()}:")
        print(f"  Final Best Fitness: {result['best_fitness']:.6f}")
        stats = result["stats_history"]
        if stats:
            initial = stats[0].best_fitness
            final = stats[-1].best_fitness
            improvement = ((final - initial) / abs(initial) * 100) if initial != 0 else 0
            print(f"  Initial Best: {initial:.6f}")
            print(f"  Fitness Improvement: {improvement:.1f}%")

            # Convergence check: did fitness plateau?
            last_10 = stats[-10:] if len(stats) >= 10 else stats
            variance = sum((s.best_fitness - last_10[-1].best_fitness) ** 2 for s in last_10) / len(
                last_10
            )
            if variance < 0.0001:
                print("  Status: CONVERGED (minimal change in last 10 generations)")
            else:
                print("  Status: STILL IMPROVING (variance in last 10 gens)")


def analyze_trait_importance(results: dict) -> None:
    """Analyze which traits evolved to extreme values.

    Args:
        results: Dictionary of results by role.
    """
    print(f"\n{'=' * 60}")
    print("TRAIT IMPORTANCE ANALYSIS")
    print(f"{'=' * 60}")

    for role, result in results.items():
        print(f"\n{role.upper()} - Optimal Genome Traits:")
        genome = result["best_genome"]

        # Sort traits by how far they are from 0.5 (neutral)
        trait_extremity = [(t, v, abs(v - 0.5)) for t, v in genome.items()]
        trait_extremity.sort(key=lambda x: x[2], reverse=True)

        for trait, value, _extremity in trait_extremity:
            if value > 0.7:
                importance = "HIGH (strongly selected for)"
            elif value < 0.3:
                importance = "LOW (selected against)"
            else:
                importance = "NEUTRAL (not strongly selected)"
            print(f"  {trait}: {value:.4f} - {importance}")


def main() -> None:
    """Run GA experiments for all three sandwich shop roles."""
    print("=" * 60)
    print("LOOPENGINE GA EXPERIMENTS")
    print("=" * 60)
    print("Configuration:")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Population Size: {POPULATION_SIZE}")
    print(f"  Selection Count: {SELECTION_COUNT}")
    print(f"  Mutation Rate: {MUTATION_RATE}")
    print(f"  Ticks per Evaluation: {TICKS_PER_EVALUATION}")

    # Define experiments
    experiments = [
        ("tom", "sandwich_maker", create_tom_schema(), tom_fitness),
        ("alex", "cashier", create_alex_schema(), alex_fitness),
        ("maria", "owner", create_maria_schema(), maria_fitness),
    ]

    # Run experiments
    results = {}
    for agent_id, role, schema, fitness_fn in experiments:
        result = run_evolution(agent_id, role, schema, fitness_fn)
        results[role] = result

    # Save results
    output_dir = Path(__file__).parent / "results"
    save_results(results, output_dir)

    # Print summary
    print_summary(results)

    # Analyze trait importance
    analyze_trait_importance(results)

    print(f"\n{'=' * 60}")
    print("EXPERIMENTS COMPLETE")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""Genetic Algorithm engine for evolving agent genomes per PRD section 4.3."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loopengine.model.genome import GenomeSchema


@dataclass
class GAStats:
    """Statistics for a single generation."""

    generation: int
    best_fitness: float
    avg_fitness: float
    min_fitness: float
    max_fitness: float
    population_size: int


@dataclass
class GAEngine:
    """Genetic algorithm engine for evolving agent genomes.

    Manages population lifecycle: initialization, evaluation, selection,
    crossover, and mutation. Tracks generation statistics and best genomes.
    """

    # Configuration
    population_size: int = 50
    selection_count: int = 10  # Top K survivors
    mutation_rate: float = 0.1  # Per-trait mutation probability
    mutation_magnitude: float = 0.05  # Std dev for Gaussian mutation
    crossover_type: str = "uniform"  # "uniform" or "blended"
    blend_alpha: float = 0.5  # For blended crossover

    # Schema defining valid traits and ranges
    schema: GenomeSchema | None = None

    # Population state
    population: list[dict[str, float]] = field(default_factory=list)
    fitness_scores: list[float] = field(default_factory=list)

    # Tracking
    generation: int = 0
    best_genome: dict[str, float] = field(default_factory=dict)
    best_fitness: float = float("-inf")
    stats_history: list[GAStats] = field(default_factory=list)

    # Fitness evaluator: callable(genome) -> float
    fitness_fn: Callable[[dict[str, float]], float] | None = None

    def initialize_population(self, schema: GenomeSchema) -> None:
        """Initialize population with random genomes within schema ranges.

        Creates N random genomes where each trait value is uniformly
        distributed within the trait's [min_val, max_val] range.

        Args:
            schema: GenomeSchema defining valid traits and their ranges.
        """
        self.schema = schema
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_genome = {}
        self.best_fitness = float("-inf")
        self.stats_history = []

        for _ in range(self.population_size):
            genome = self._create_random_genome(schema)
            self.population.append(genome)

    def _create_random_genome(self, schema: GenomeSchema) -> dict[str, float]:
        """Create a random genome with trait values in schema ranges.

        Args:
            schema: GenomeSchema defining traits and ranges.

        Returns:
            dict[str, float]: Genome with random trait values.
        """
        genome: dict[str, float] = {}
        for trait_name, trait in schema.traits.items():
            value = random.uniform(trait.min_val, trait.max_val)
            genome[trait_name] = value
        return genome

    def evaluate_population(
        self, fitness_fn: Callable[[dict[str, float]], float] | None = None
    ) -> None:
        """Evaluate fitness for all genomes in the population.

        Args:
            fitness_fn: Optional fitness function. If not provided,
                        uses the stored self.fitness_fn.
        """
        if fitness_fn is not None:
            self.fitness_fn = fitness_fn

        if self.fitness_fn is None:
            msg = "No fitness function provided"
            raise ValueError(msg)

        self.fitness_scores = [self.fitness_fn(genome) for genome in self.population]

        # Update best genome tracking
        for i, fitness in enumerate(self.fitness_scores):
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = self.population[i].copy()

    def select(self) -> list[dict[str, float]]:
        """Select top K genomes for survival.

        Uses rank selection: sorts population by fitness and takes
        the top selection_count genomes.

        Returns:
            list[dict[str, float]]: Selected survivor genomes.
        """
        if not self.fitness_scores:
            msg = "Population not evaluated - call evaluate_population first"
            raise ValueError(msg)

        # Sort by fitness (descending)
        indexed = list(enumerate(self.fitness_scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        # Select top K
        survivors = [self.population[i].copy() for i, _ in indexed[: self.selection_count]]
        return survivors

    def crossover(self, parent1: dict[str, float], parent2: dict[str, float]) -> dict[str, float]:
        """Produce offspring by combining two parent genomes.

        Supports two crossover strategies:
        - uniform: each trait independently chosen from one parent
        - blended: each trait is weighted average of parents

        Args:
            parent1: First parent genome.
            parent2: Second parent genome.

        Returns:
            dict[str, float]: Offspring genome.
        """
        offspring: dict[str, float] = {}

        # Get all trait names from both parents
        all_traits = set(parent1.keys()) | set(parent2.keys())

        for trait in all_traits:
            val1 = parent1.get(trait, 0.5)
            val2 = parent2.get(trait, 0.5)

            if self.crossover_type == "uniform":
                # Randomly choose from one parent
                offspring[trait] = val1 if random.random() < 0.5 else val2
            else:  # blended
                # Weighted average with some variation
                weight = random.uniform(0.0, 1.0)
                offspring[trait] = weight * val1 + (1 - weight) * val2

        return offspring

    def mutate(self, genome: dict[str, float]) -> dict[str, float]:
        """Apply random mutations to a genome.

        For each trait, with probability mutation_rate, perturb the value
        by Gaussian noise with std dev mutation_magnitude. Values are
        clamped to schema ranges.

        Args:
            genome: Genome to mutate.

        Returns:
            dict[str, float]: Mutated genome (may be same object).
        """
        mutated = genome.copy()

        for trait_name, value in mutated.items():
            if random.random() < self.mutation_rate:
                # Apply Gaussian perturbation
                delta = random.gauss(0, self.mutation_magnitude)
                new_value = value + delta

                # Clamp to schema range if available
                if self.schema and trait_name in self.schema.traits:
                    trait = self.schema.traits[trait_name]
                    new_value = max(trait.min_val, min(trait.max_val, new_value))
                else:
                    # Default clamp to [0, 1]
                    new_value = max(0.0, min(1.0, new_value))

                mutated[trait_name] = new_value

        return mutated

    def run_generation(
        self, fitness_fn: Callable[[dict[str, float]], float] | None = None
    ) -> GAStats:
        """Execute one full generation cycle: evaluate → select → crossover → mutate.

        Args:
            fitness_fn: Optional fitness function. If not provided,
                        uses the stored self.fitness_fn.

        Returns:
            GAStats: Statistics for this generation.
        """
        # Evaluate current population
        self.evaluate_population(fitness_fn)

        # Record stats before evolving
        stats = GAStats(
            generation=self.generation,
            best_fitness=max(self.fitness_scores),
            avg_fitness=sum(self.fitness_scores) / len(self.fitness_scores),
            min_fitness=min(self.fitness_scores),
            max_fitness=max(self.fitness_scores),
            population_size=len(self.population),
        )
        self.stats_history.append(stats)

        # Select survivors
        survivors = self.select()

        # Generate offspring via crossover and mutation
        offspring: list[dict[str, float]] = []
        offspring_needed = self.population_size - len(survivors)

        while len(offspring) < offspring_needed:
            # Select two random parents from survivors
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)

            # Create offspring via crossover
            child = self.crossover(parent1, parent2)

            # Apply mutation
            child = self.mutate(child)

            offspring.append(child)

        # New population = survivors + offspring
        self.population = survivors + offspring
        self.fitness_scores = []  # Reset for next evaluation
        self.generation += 1

        return stats

    def run(
        self,
        generations: int,
        fitness_fn: Callable[[dict[str, float]], float] | None = None,
    ) -> dict[str, Any]:
        """Run the GA for multiple generations.

        Args:
            generations: Number of generations to run.
            fitness_fn: Optional fitness function.

        Returns:
            dict: Results containing best_genome, best_fitness, and stats_history.
        """
        for _ in range(generations):
            self.run_generation(fitness_fn)

        return {
            "best_genome": self.best_genome,
            "best_fitness": self.best_fitness,
            "generations_run": generations,
            "stats_history": self.stats_history,
        }

    def get_population_stats(self) -> dict[str, Any]:
        """Get current population statistics.

        Returns:
            dict: Current generation, population size, best fitness, etc.
        """
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_fitness,
            "best_genome": self.best_genome,
            "fitness_evaluated": len(self.fitness_scores) > 0,
            "stats_history_length": len(self.stats_history),
        }

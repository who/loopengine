"""Tests for the genetic algorithm engine (loopengine.engine.ga)."""

from __future__ import annotations

import pytest

from loopengine.engine.fitness import alex_fitness, maria_fitness, tom_fitness
from loopengine.engine.ga import FitnessEvaluator, GAEngine, GAStats, evaluate_fitness
from loopengine.model.genome import GenomeSchema, GenomeTrait


@pytest.fixture
def sandwich_maker_schema() -> GenomeSchema:
    """Create a sandwich_maker GenomeSchema for testing."""
    return GenomeSchema(
        role="sandwich_maker",
        traits={
            "speed": GenomeTrait(
                name="speed",
                description="How fast the agent works",
                min_val=0.0,
                max_val=1.0,
                category="physical",
            ),
            "consistency": GenomeTrait(
                name="consistency",
                description="Quality consistency of output",
                min_val=0.0,
                max_val=1.0,
                category="skill",
            ),
            "ingredient_intuition": GenomeTrait(
                name="ingredient_intuition",
                description="Ability to substitute ingredients",
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
                description="Efficiency with materials",
                min_val=0.0,
                max_val=1.0,
                category="skill",
            ),
        },
    )


@pytest.fixture
def simple_fitness_fn():
    """Simple fitness function that sums trait values."""

    def fitness(genome: dict[str, float]) -> float:
        return sum(genome.values())

    return fitness


@pytest.fixture
def ga_engine() -> GAEngine:
    """Create a GA engine with default settings."""
    return GAEngine(population_size=10, selection_count=3)


class TestGAEngineInit:
    """Test GA engine initialization."""

    def test_ga_init_creates_correct_population_size(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test that initialize_population creates N genomes."""
        ga_engine.initialize_population(sandwich_maker_schema)
        assert len(ga_engine.population) == 10

    def test_ga_init_genomes_have_all_traits(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test that genomes contain all schema traits."""
        ga_engine.initialize_population(sandwich_maker_schema)
        expected_traits = set(sandwich_maker_schema.traits.keys())

        for genome in ga_engine.population:
            assert set(genome.keys()) == expected_traits

    def test_ga_init_trait_values_in_range(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test that all trait values are in [0, 1] range."""
        ga_engine.initialize_population(sandwich_maker_schema)

        for genome in ga_engine.population:
            for trait_name, value in genome.items():
                trait = sandwich_maker_schema.traits[trait_name]
                assert trait.min_val <= value <= trait.max_val, (
                    f"Trait {trait_name} value {value} outside range "
                    f"[{trait.min_val}, {trait.max_val}]"
                )

    def test_ga_init_resets_state(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test that initialize_population resets tracking state."""
        ga_engine.best_fitness = 100.0
        ga_engine.generation = 50
        ga_engine.best_genome = {"test": 0.5}

        ga_engine.initialize_population(sandwich_maker_schema)

        assert ga_engine.generation == 0
        assert ga_engine.best_fitness == float("-inf")
        assert ga_engine.best_genome == {}
        assert ga_engine.stats_history == []


class TestGAEngineEvaluation:
    """Test fitness evaluation."""

    def test_evaluate_population(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test that evaluate_population computes fitness for all genomes."""
        ga_engine.initialize_population(sandwich_maker_schema)
        ga_engine.evaluate_population(simple_fitness_fn)

        assert len(ga_engine.fitness_scores) == len(ga_engine.population)
        assert all(isinstance(f, float) for f in ga_engine.fitness_scores)

    def test_evaluate_updates_best_genome(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test that evaluation tracks best genome."""
        ga_engine.initialize_population(sandwich_maker_schema)
        ga_engine.evaluate_population(simple_fitness_fn)

        assert ga_engine.best_fitness > float("-inf")
        assert ga_engine.best_genome != {}

        # Best fitness should match the genome
        expected_fitness = simple_fitness_fn(ga_engine.best_genome)
        assert ga_engine.best_fitness == expected_fitness

    def test_evaluate_without_fitness_fn_raises(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test that evaluating without fitness function raises error."""
        ga_engine.initialize_population(sandwich_maker_schema)

        with pytest.raises(ValueError, match="No fitness function"):
            ga_engine.evaluate_population()


class TestGAEngineSelection:
    """Test selection mechanism."""

    def test_select_returns_top_k(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test that select returns top K genomes."""
        ga_engine.initialize_population(sandwich_maker_schema)
        ga_engine.evaluate_population(simple_fitness_fn)

        survivors = ga_engine.select()

        assert len(survivors) == ga_engine.selection_count

    def test_select_returns_best_genomes(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test that selection picks highest fitness genomes."""
        ga_engine.initialize_population(sandwich_maker_schema)
        ga_engine.evaluate_population(simple_fitness_fn)

        survivors = ga_engine.select()
        survivor_fitnesses = [simple_fitness_fn(g) for g in survivors]

        # Get sorted fitnesses from original population
        sorted_fitnesses = sorted(ga_engine.fitness_scores, reverse=True)
        expected_top = sorted_fitnesses[: ga_engine.selection_count]

        assert sorted(survivor_fitnesses, reverse=True) == sorted(expected_top, reverse=True)

    def test_select_without_evaluation_raises(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test that selecting without evaluation raises error."""
        ga_engine.initialize_population(sandwich_maker_schema)

        with pytest.raises(ValueError, match="not evaluated"):
            ga_engine.select()


class TestGAEngineSelectionAdvanced:
    """Test advanced selection mechanisms (tournament, ratio)."""

    def test_selection_ratio_calculates_k(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test selection_ratio computes correct survivor count."""
        ga = GAEngine(population_size=20, selection_count=None, selection_ratio=0.3)
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()

        # 0.3 * 20 = 6 survivors
        assert len(survivors) == 6

    def test_selection_ratio_minimum_one(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test selection_ratio returns at least 1 survivor."""
        ga = GAEngine(population_size=10, selection_count=None, selection_ratio=0.01)
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()

        # Even with tiny ratio, at least 1 should survive
        assert len(survivors) >= 1

    def test_selection_count_overrides_ratio(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test selection_count takes precedence over selection_ratio."""
        ga = GAEngine(population_size=20, selection_count=5, selection_ratio=0.5)
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()

        # selection_count should override ratio (5 not 10)
        assert len(survivors) == 5

    def test_tournament_selection_returns_k(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test tournament selection returns correct number of survivors."""
        ga = GAEngine(
            population_size=20, selection_count=5, selection_type="tournament", tournament_size=3
        )
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()

        assert len(survivors) == 5

    def test_tournament_selection_produces_valid_genomes(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test tournament selection returns genomes from population."""
        ga = GAEngine(
            population_size=10, selection_count=3, selection_type="tournament", tournament_size=2
        )
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()

        # Each survivor should have all expected traits
        expected_traits = set(sandwich_maker_schema.traits.keys())
        for survivor in survivors:
            assert set(survivor.keys()) == expected_traits

    def test_tournament_selection_sorted_by_fitness(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test tournament selection returns survivors sorted by fitness."""
        ga = GAEngine(
            population_size=20, selection_count=5, selection_type="tournament", tournament_size=4
        )
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()
        survivor_fitnesses = [simple_fitness_fn(g) for g in survivors]

        # Should be sorted descending
        assert survivor_fitnesses == sorted(survivor_fitnesses, reverse=True)

    def test_rank_selection_sorted_by_fitness(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test rank selection returns survivors sorted by fitness (best first)."""
        ga = GAEngine(population_size=20, selection_count=5, selection_type="rank")
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()
        survivor_fitnesses = [simple_fitness_fn(g) for g in survivors]

        # Should be sorted descending
        assert survivor_fitnesses == sorted(survivor_fitnesses, reverse=True)

    def test_tournament_with_ratio(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test tournament selection works with selection_ratio."""
        ga = GAEngine(
            population_size=20,
            selection_count=None,
            selection_ratio=0.2,
            selection_type="tournament",
            tournament_size=3,
        )
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()

        # 0.2 * 20 = 4 survivors
        assert len(survivors) == 4

    def test_handles_ties_gracefully(
        self,
        sandwich_maker_schema: GenomeSchema,
    ) -> None:
        """Test selection handles fitness ties consistently."""
        ga = GAEngine(population_size=5, selection_count=3, selection_type="rank")
        ga.initialize_population(sandwich_maker_schema)

        # Set all fitness scores to same value (ties)
        def constant_fitness(genome: dict[str, float]) -> float:
            return 1.0

        ga.evaluate_population(constant_fitness)
        survivors = ga.select()

        # Should still return 3 survivors even with ties
        assert len(survivors) == 3

    def test_tournament_handles_small_population(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test tournament selection works when tournament_size > population."""
        ga = GAEngine(
            population_size=3, selection_count=2, selection_type="tournament", tournament_size=10
        )
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        # Should not raise error
        survivors = ga.select()
        assert len(survivors) == 2

    def test_default_selection_when_neither_count_nor_ratio(
        self,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test default behavior when neither selection_count nor ratio is set."""
        ga = GAEngine(population_size=10, selection_count=None, selection_ratio=None)
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()

        # Default is 20% of population = 2
        assert len(survivors) == 2


class TestGAEngineCrossover:
    """Test crossover operators."""

    def test_uniform_crossover_produces_valid_genome(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test uniform crossover produces genome with all traits."""
        ga_engine.crossover_type = "uniform"
        ga_engine.initialize_population(sandwich_maker_schema)

        parent1 = ga_engine.population[0]
        parent2 = ga_engine.population[1]

        offspring = ga_engine.crossover(parent1, parent2)

        assert set(offspring.keys()) == set(parent1.keys())

    def test_uniform_crossover_values_from_parents(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test uniform crossover values come from one parent."""
        ga_engine.crossover_type = "uniform"
        ga_engine.initialize_population(sandwich_maker_schema)

        parent1 = ga_engine.population[0]
        parent2 = ga_engine.population[1]

        offspring = ga_engine.crossover(parent1, parent2)

        for trait, value in offspring.items():
            assert value == parent1[trait] or value == parent2[trait]

    def test_blended_crossover_produces_valid_genome(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test blended crossover produces genome with all traits."""
        ga_engine.crossover_type = "blended"
        ga_engine.initialize_population(sandwich_maker_schema)

        parent1 = ga_engine.population[0]
        parent2 = ga_engine.population[1]

        offspring = ga_engine.crossover(parent1, parent2)

        assert set(offspring.keys()) == set(parent1.keys())

    def test_blended_crossover_values_between_parents(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test blended crossover produces intermediate values."""
        ga_engine.crossover_type = "blended"
        ga_engine.initialize_population(sandwich_maker_schema)

        parent1 = ga_engine.population[0]
        parent2 = ga_engine.population[1]

        offspring = ga_engine.crossover(parent1, parent2)

        for trait, value in offspring.items():
            min_val = min(parent1[trait], parent2[trait])
            max_val = max(parent1[trait], parent2[trait])
            assert min_val <= value <= max_val


class TestGAEngineMutation:
    """Test mutation operators."""

    def test_mutate_preserves_traits(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test mutation preserves all trait keys."""
        ga_engine.initialize_population(sandwich_maker_schema)
        genome = ga_engine.population[0]

        mutated = ga_engine.mutate(genome)

        assert set(mutated.keys()) == set(genome.keys())

    def test_mutate_values_in_range(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test mutated values stay within schema ranges."""
        ga_engine.mutation_rate = 1.0  # Mutate everything
        ga_engine.mutation_magnitude = 0.5  # Large mutations
        ga_engine.initialize_population(sandwich_maker_schema)

        genome = ga_engine.population[0]
        mutated = ga_engine.mutate(genome)

        for trait_name, value in mutated.items():
            trait = sandwich_maker_schema.traits[trait_name]
            assert trait.min_val <= value <= trait.max_val

    def test_no_mutation_with_zero_rate(
        self, ga_engine: GAEngine, sandwich_maker_schema: GenomeSchema
    ) -> None:
        """Test no mutation occurs with zero mutation rate."""
        ga_engine.mutation_rate = 0.0
        ga_engine.initialize_population(sandwich_maker_schema)

        genome = ga_engine.population[0].copy()
        mutated = ga_engine.mutate(genome.copy())

        assert mutated == genome


class TestGAEngineGeneration:
    """Test full generation cycle."""

    def test_run_generation_advances_generation(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test run_generation increments generation counter."""
        ga_engine.initialize_population(sandwich_maker_schema)

        assert ga_engine.generation == 0
        ga_engine.run_generation(simple_fitness_fn)
        assert ga_engine.generation == 1

    def test_run_generation_returns_stats(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test run_generation returns GAStats."""
        ga_engine.initialize_population(sandwich_maker_schema)

        stats = ga_engine.run_generation(simple_fitness_fn)

        assert isinstance(stats, GAStats)
        assert stats.generation == 0
        assert stats.population_size == ga_engine.population_size

    def test_run_generation_updates_stats_history(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test run_generation appends to stats_history."""
        ga_engine.initialize_population(sandwich_maker_schema)

        ga_engine.run_generation(simple_fitness_fn)
        ga_engine.run_generation(simple_fitness_fn)

        assert len(ga_engine.stats_history) == 2

    def test_run_generation_maintains_population_size(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test population size stays constant across generations."""
        ga_engine.initialize_population(sandwich_maker_schema)

        initial_size = len(ga_engine.population)
        ga_engine.run_generation(simple_fitness_fn)

        assert len(ga_engine.population) == initial_size


class TestGAEngineRun:
    """Test multi-generation run."""

    def test_run_executes_all_generations(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test run executes specified number of generations."""
        ga_engine.initialize_population(sandwich_maker_schema)

        result = ga_engine.run(generations=5, fitness_fn=simple_fitness_fn)

        assert result["generations_run"] == 5
        assert ga_engine.generation == 5
        assert len(ga_engine.stats_history) == 5

    def test_run_returns_best_genome(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test run returns best genome found."""
        ga_engine.initialize_population(sandwich_maker_schema)

        result = ga_engine.run(generations=10, fitness_fn=simple_fitness_fn)

        assert "best_genome" in result
        assert "best_fitness" in result
        assert result["best_fitness"] > float("-inf")

    def test_fitness_improves_over_generations(self, sandwich_maker_schema: GenomeSchema) -> None:
        """Test fitness improves over multiple generations (statistical)."""
        # Use larger population for more reliable improvement
        ga_engine = GAEngine(population_size=50, selection_count=10)
        ga_engine.initialize_population(sandwich_maker_schema)

        # Fitness function that rewards high trait values
        def fitness(genome: dict[str, float]) -> float:
            return sum(genome.values())

        # Run for 50 generations
        ga_engine.run(generations=50, fitness_fn=fitness)

        # Compare early vs late generation fitness
        early_stats = ga_engine.stats_history[0]
        late_stats = ga_engine.stats_history[-1]

        # Best fitness should generally improve
        assert late_stats.best_fitness >= early_stats.best_fitness


class TestGAEngineStats:
    """Test statistics tracking."""

    def test_get_population_stats(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test get_population_stats returns expected fields."""
        ga_engine.initialize_population(sandwich_maker_schema)
        ga_engine.run_generation(simple_fitness_fn)

        stats = ga_engine.get_population_stats()

        assert "generation" in stats
        assert "population_size" in stats
        assert "best_fitness" in stats
        assert "best_genome" in stats
        assert stats["generation"] == 1
        assert stats["population_size"] == ga_engine.population_size

    def test_ga_stats_dataclass(
        self,
        ga_engine: GAEngine,
        sandwich_maker_schema: GenomeSchema,
        simple_fitness_fn,
    ) -> None:
        """Test GAStats dataclass has all expected fields."""
        ga_engine.initialize_population(sandwich_maker_schema)
        stats = ga_engine.run_generation(simple_fitness_fn)

        assert hasattr(stats, "generation")
        assert hasattr(stats, "best_fitness")
        assert hasattr(stats, "avg_fitness")
        assert hasattr(stats, "min_fitness")
        assert hasattr(stats, "max_fitness")
        assert hasattr(stats, "population_size")

        # Fitness ordering should be correct
        assert stats.min_fitness <= stats.avg_fitness <= stats.max_fitness


class TestGAEngineConfiguration:
    """Test configuration options."""

    def test_configurable_population_size(self, sandwich_maker_schema: GenomeSchema) -> None:
        """Test population size is configurable."""
        for size in [5, 20, 100]:
            ga = GAEngine(population_size=size)
            ga.initialize_population(sandwich_maker_schema)
            assert len(ga.population) == size

    def test_configurable_selection_count(
        self, sandwich_maker_schema: GenomeSchema, simple_fitness_fn
    ) -> None:
        """Test selection count is configurable."""
        ga = GAEngine(population_size=20, selection_count=5)
        ga.initialize_population(sandwich_maker_schema)
        ga.evaluate_population(simple_fitness_fn)

        survivors = ga.select()
        assert len(survivors) == 5

    def test_configurable_mutation_rate(self, sandwich_maker_schema: GenomeSchema) -> None:
        """Test mutation rate affects mutation frequency."""
        ga = GAEngine(mutation_rate=1.0, mutation_magnitude=0.5)
        ga.initialize_population(sandwich_maker_schema)

        genome = ga.population[0].copy()
        mutated = ga.mutate(genome.copy())

        # With 100% mutation rate and high magnitude, values should differ
        differences = sum(1 for t in genome if abs(genome[t] - mutated[t]) > 0.001)
        assert differences > 0  # At least some traits mutated


class TestFitnessEvaluation:
    """Test fitness evaluation framework."""

    @pytest.fixture
    def sandwich_world(self):
        """Create a sandwich shop world for testing."""
        from loopengine.corpora.sandwich_shop import create_world

        return create_world()

    def test_evaluate_fitness_runs_simulation(self, sandwich_world) -> None:
        """Test evaluate_fitness runs simulation for specified ticks."""
        genome = {"speed": 0.7, "consistency": 0.8, "ingredient_intuition": 0.6}

        def counting_fitness(world, agent_id) -> float:
            return float(world.tick)

        fitness = evaluate_fitness(
            genome=genome,
            world_template=sandwich_world,
            target_agent_id="tom",
            ticks=100,
            fitness_fn=counting_fitness,
            seed=42,
        )

        assert fitness == 100.0

    def test_evaluate_fitness_applies_genome(self, sandwich_world) -> None:
        """Test evaluate_fitness applies genome to target agent."""
        test_genome = {"speed": 0.99, "consistency": 0.01}

        def genome_check_fitness(world, agent_id) -> float:
            agent_genome = world.agents[agent_id].genome
            return agent_genome.get("speed", 0) - agent_genome.get("consistency", 0)

        fitness = evaluate_fitness(
            genome=test_genome,
            world_template=sandwich_world,
            target_agent_id="tom",
            ticks=1,
            fitness_fn=genome_check_fitness,
        )

        # speed - consistency = 0.99 - 0.01 = 0.98
        assert abs(fitness - 0.98) < 0.001

    def test_evaluate_fitness_isolates_world(self, sandwich_world) -> None:
        """Test evaluate_fitness does not modify original world."""
        original_tick = sandwich_world.tick
        original_genome = sandwich_world.agents["tom"].genome.copy()

        evaluate_fitness(
            genome={"speed": 0.1, "consistency": 0.1},
            world_template=sandwich_world,
            target_agent_id="tom",
            ticks=50,
            fitness_fn=lambda w, a: 1.0,
        )

        # Original world should be unchanged
        assert sandwich_world.tick == original_tick
        assert sandwich_world.agents["tom"].genome == original_genome

    def test_evaluate_fitness_deterministic_with_seed(self, sandwich_world) -> None:
        """Test evaluate_fitness is deterministic given seed."""
        genome = {"speed": 0.7, "consistency": 0.8}

        def served_fitness(world, agent_id) -> float:
            return float(
                sum(a.internal_state.get("served_count", 0) for a in world.agents.values())
            )

        fitness1 = evaluate_fitness(
            genome=genome,
            world_template=sandwich_world,
            target_agent_id="tom",
            ticks=200,
            fitness_fn=served_fitness,
            seed=42,
        )

        fitness2 = evaluate_fitness(
            genome=genome,
            world_template=sandwich_world,
            target_agent_id="tom",
            ticks=200,
            fitness_fn=served_fitness,
            seed=42,
        )

        assert fitness1 == fitness2

    def test_evaluate_fitness_returns_float(self, sandwich_world) -> None:
        """Test evaluate_fitness returns scalar float."""
        fitness = evaluate_fitness(
            genome={"speed": 0.5},
            world_template=sandwich_world,
            target_agent_id="tom",
            ticks=10,
            fitness_fn=lambda w, a: 42.5,
        )

        assert isinstance(fitness, float)
        assert fitness == 42.5

    def test_evaluate_fitness_default_fitness_fn(self, sandwich_world) -> None:
        """Test evaluate_fitness uses default fitness function when none provided."""
        fitness = evaluate_fitness(
            genome={"speed": 0.7, "consistency": 0.8},
            world_template=sandwich_world,
            target_agent_id="tom",
            ticks=500,
            seed=42,
        )

        # Default fitness is total served, should be >= 0
        assert isinstance(fitness, float)
        assert fitness >= 0.0

    def test_evaluate_fitness_missing_agent_raises(self, sandwich_world) -> None:
        """Test evaluate_fitness raises error for missing agent."""
        with pytest.raises(ValueError, match="not found"):
            evaluate_fitness(
                genome={"speed": 0.5},
                world_template=sandwich_world,
                target_agent_id="nonexistent_agent",
                ticks=10,
                fitness_fn=lambda w, a: 1.0,
            )


class TestFitnessEvaluatorClass:
    """Test FitnessEvaluator class directly."""

    @pytest.fixture
    def sandwich_world(self):
        """Create a sandwich shop world for testing."""
        from loopengine.corpora.sandwich_shop import create_world

        return create_world()

    def test_evaluator_requires_target_agent_id(self, sandwich_world) -> None:
        """Test FitnessEvaluator raises error without target_agent_id."""
        evaluator = FitnessEvaluator(fitness_fn=lambda w, a: 1.0)

        with pytest.raises(ValueError, match="target_agent_id"):
            evaluator.evaluate({"speed": 0.5}, sandwich_world)

    def test_evaluator_requires_fitness_fn(self, sandwich_world) -> None:
        """Test FitnessEvaluator raises error without fitness_fn."""
        evaluator = FitnessEvaluator(target_agent_id="tom")

        with pytest.raises(ValueError, match="fitness_fn"):
            evaluator.evaluate({"speed": 0.5}, sandwich_world)

    def test_evaluator_configurable_ticks(self, sandwich_world) -> None:
        """Test FitnessEvaluator uses configurable tick count."""
        ticks_run: list[int] = []

        def track_ticks(world, agent_id) -> float:
            ticks_run.append(world.tick)
            return 1.0

        evaluator = FitnessEvaluator(
            target_agent_id="tom",
            ticks=75,
            fitness_fn=track_ticks,
        )

        evaluator.evaluate({"speed": 0.5}, sandwich_world)

        assert ticks_run[0] == 75

    def test_evaluator_captures_history(self, sandwich_world) -> None:
        """Test FitnessEvaluator runs simulation that updates agent state."""

        def check_internal_state(world, agent_id) -> float:
            # Alex should have processed some customers (has internal_state changes)
            alex = world.agents["alex"]
            return float(len(alex.internal_state))

        evaluator = FitnessEvaluator(
            target_agent_id="tom",
            ticks=200,
            fitness_fn=check_internal_state,
            seed=42,
        )

        fitness = evaluator.evaluate({"speed": 0.8}, sandwich_world)

        # Alex should have internal state (waiting_customers, possibly served_count)
        assert fitness > 0


class TestTomFitness:
    """Test Tom's fitness function per PRD section 9.6."""

    @pytest.fixture
    def sandwich_world(self):
        """Create a sandwich shop world for testing."""
        from loopengine.corpora.sandwich_shop import create_world

        return create_world()

    def test_tom_fitness_counts_sandwiches_completed(self, sandwich_world) -> None:
        """Test that tom_fitness counts sandwiches_completed from history."""
        # Set up Tom's internal state with known values
        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 10
        tom.internal_state["quality_scores"] = [0.8] * 10
        tom.internal_state["waste_count"] = 0
        tom.internal_state["ingredients_used"] = 10
        sandwich_world.tick = 100

        # Alex's queue depth
        alex = sandwich_world.agents["alex"]
        alex.internal_state["max_queue_depth"] = 0

        fitness = tom_fitness(sandwich_world, "tom")

        # throughput = 10/100 = 0.1, avg_consistency = 0.8
        # fitness = 0.1 * 0.8 - 0 - 0 = 0.08
        assert abs(fitness - 0.08) < 0.001

    def test_tom_fitness_computes_average_consistency_score(self, sandwich_world) -> None:
        """Test that tom_fitness computes average_consistency_score."""
        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 5
        tom.internal_state["quality_scores"] = [0.6, 0.7, 0.8, 0.9, 1.0]  # avg = 0.8
        tom.internal_state["waste_count"] = 0
        tom.internal_state["ingredients_used"] = 5
        sandwich_world.tick = 100

        alex = sandwich_world.agents["alex"]
        alex.internal_state["max_queue_depth"] = 0

        fitness = tom_fitness(sandwich_world, "tom")

        # throughput = 5/100 = 0.05, avg_consistency = 0.8
        # fitness = 0.05 * 0.8 - 0 - 0 = 0.04
        assert abs(fitness - 0.04) < 0.001

    def test_tom_fitness_tracks_waste_particles(self, sandwich_world) -> None:
        """Test that tom_fitness tracks waste_particles."""
        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 10
        tom.internal_state["quality_scores"] = [1.0] * 10
        tom.internal_state["waste_count"] = 5
        tom.internal_state["ingredients_used"] = 15  # 10 sandwiches + 5 waste
        sandwich_world.tick = 100

        alex = sandwich_world.agents["alex"]
        alex.internal_state["max_queue_depth"] = 0

        fitness = tom_fitness(sandwich_world, "tom")

        # throughput = 10/100 = 0.1, avg_consistency = 1.0
        # waste_ratio = 5/15 = 0.333...
        # fitness = 0.1 * 1.0 - 0.333... - 0 ≈ -0.233
        expected = 0.1 * 1.0 - (5 / 15)
        assert abs(fitness - expected) < 0.001

    def test_tom_fitness_measures_max_queue_depth(self, sandwich_world) -> None:
        """Test that tom_fitness measures max_queue_depth."""
        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 10
        tom.internal_state["quality_scores"] = [1.0] * 10
        tom.internal_state["waste_count"] = 0
        tom.internal_state["ingredients_used"] = 10
        sandwich_world.tick = 100

        alex = sandwich_world.agents["alex"]
        alex.internal_state["max_queue_depth"] = 5

        fitness = tom_fitness(sandwich_world, "tom")

        # throughput = 10/100 = 0.1, avg_consistency = 1.0
        # queue_penalty = 5 * 0.1 = 0.5
        # fitness = 0.1 * 1.0 - 0 - 0.5 = -0.4
        expected = 0.1 * 1.0 - 0 - (5 * 0.1)
        assert abs(fitness - expected) < 0.001

    def test_tom_fitness_returns_scalar(self, sandwich_world) -> None:
        """Test that tom_fitness returns scalar fitness."""
        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 5
        tom.internal_state["quality_scores"] = [0.8] * 5
        tom.internal_state["waste_count"] = 1
        tom.internal_state["ingredients_used"] = 6
        sandwich_world.tick = 50

        alex = sandwich_world.agents["alex"]
        alex.internal_state["max_queue_depth"] = 2

        fitness = tom_fitness(sandwich_world, "tom")

        assert isinstance(fitness, float)

    def test_tom_fitness_higher_throughput_higher_fitness(self, sandwich_world) -> None:
        """Test that higher throughput = higher fitness."""
        # Low throughput scenario
        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 5
        tom.internal_state["quality_scores"] = [0.8] * 5
        tom.internal_state["waste_count"] = 0
        tom.internal_state["ingredients_used"] = 5
        sandwich_world.tick = 100

        alex = sandwich_world.agents["alex"]
        alex.internal_state["max_queue_depth"] = 0

        low_throughput_fitness = tom_fitness(sandwich_world, "tom")

        # High throughput scenario
        tom.internal_state["sandwiches_completed"] = 20
        tom.internal_state["quality_scores"] = [0.8] * 20
        tom.internal_state["ingredients_used"] = 20

        high_throughput_fitness = tom_fitness(sandwich_world, "tom")

        assert high_throughput_fitness > low_throughput_fitness

    def test_tom_fitness_high_waste_lower_fitness(self, sandwich_world) -> None:
        """Test that high waste = lower fitness."""
        # Low waste scenario
        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 10
        tom.internal_state["quality_scores"] = [0.8] * 10
        tom.internal_state["waste_count"] = 0
        tom.internal_state["ingredients_used"] = 10
        sandwich_world.tick = 100

        alex = sandwich_world.agents["alex"]
        alex.internal_state["max_queue_depth"] = 0

        low_waste_fitness = tom_fitness(sandwich_world, "tom")

        # High waste scenario
        tom.internal_state["waste_count"] = 10
        tom.internal_state["ingredients_used"] = 20  # 10 sandwiches + 10 waste

        high_waste_fitness = tom_fitness(sandwich_world, "tom")

        assert low_waste_fitness > high_waste_fitness

    def test_tom_fitness_handles_zero_ticks(self, sandwich_world) -> None:
        """Test that tom_fitness handles zero ticks gracefully."""
        sandwich_world.tick = 0

        fitness = tom_fitness(sandwich_world, "tom")

        assert fitness == 0.0

    def test_tom_fitness_handles_empty_internal_state(self, sandwich_world) -> None:
        """Test that tom_fitness handles empty internal state."""
        tom = sandwich_world.agents["tom"]
        tom.internal_state.clear()
        sandwich_world.tick = 100

        alex = sandwich_world.agents["alex"]
        alex.internal_state.clear()

        fitness = tom_fitness(sandwich_world, "tom")

        # All metrics default to 0
        assert fitness == 0.0

    def test_tom_fitness_raises_for_missing_agent(self, sandwich_world) -> None:
        """Test that tom_fitness raises error for missing agent."""
        with pytest.raises(ValueError, match="not found"):
            tom_fitness(sandwich_world, "nonexistent_agent")

    def test_tom_fitness_integration_with_simulation(self, sandwich_world) -> None:
        """Test tom_fitness with actual simulation run."""
        # Run simulation for a bit
        fitness = evaluate_fitness(
            genome={"speed": 0.7, "consistency": 0.8, "waste_minimization": 0.5},
            world_template=sandwich_world,
            target_agent_id="tom",
            ticks=500,
            fitness_fn=tom_fitness,
            seed=42,
        )

        # Fitness should be non-zero after simulation
        assert isinstance(fitness, float)
        # With realistic simulation, fitness will depend on sandwich production
        # Could be negative due to queue penalties, but should be a valid number

    def test_tom_fitness_formula_components(self, sandwich_world) -> None:
        """Test that fitness formula combines all components correctly."""
        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 50
        tom.internal_state["quality_scores"] = [0.9] * 50
        tom.internal_state["waste_count"] = 5
        tom.internal_state["ingredients_used"] = 55  # 50 + 5 waste
        sandwich_world.tick = 1000

        alex = sandwich_world.agents["alex"]
        alex.internal_state["max_queue_depth"] = 3

        fitness = tom_fitness(sandwich_world, "tom")

        # Manual calculation:
        throughput = 50 / 1000  # = 0.05
        avg_consistency = 0.9
        waste_ratio = 5 / 55  # ≈ 0.0909
        queue_penalty = 3 * 0.1  # = 0.3

        expected = (throughput * avg_consistency) - waste_ratio - queue_penalty
        # = (0.05 * 0.9) - 0.0909 - 0.3 = 0.045 - 0.0909 - 0.3 ≈ -0.346

        assert abs(fitness - expected) < 0.001


class TestAlexFitness:
    """Test Alex's fitness function per PRD section 9.6."""

    @pytest.fixture
    def sandwich_world(self):
        """Create a sandwich shop world for testing."""
        from loopengine.corpora.sandwich_shop import create_world

        return create_world()

    def test_alex_fitness_counts_customers_served(self, sandwich_world) -> None:
        """Test that alex_fitness counts customers_served from history."""
        alex = sandwich_world.agents["alex"]
        alex.internal_state["customers_served"] = 20
        alex.internal_state["total_orders"] = 20
        alex.internal_state["accurate_orders"] = 20
        alex.internal_state["customer_wait_times"] = [0] * 20
        alex.internal_state["upsell_count"] = 0
        sandwich_world.tick = 100

        fitness = alex_fitness(sandwich_world, "alex")

        # throughput = 20/100 = 0.2, accuracy = 20/20 = 1.0
        # wait_penalty = 0 * 0.05 = 0, upsell_bonus = 0 * 0.1 = 0
        # fitness = 0.2 * 1.0 - 0 + 0 = 0.2
        assert abs(fitness - 0.2) < 0.001

    def test_alex_fitness_computes_order_accuracy_rate(self, sandwich_world) -> None:
        """Test that alex_fitness computes order_accuracy_rate."""
        alex = sandwich_world.agents["alex"]
        alex.internal_state["customers_served"] = 10
        alex.internal_state["total_orders"] = 10
        alex.internal_state["accurate_orders"] = 8  # 80% accuracy
        alex.internal_state["customer_wait_times"] = [0] * 10
        alex.internal_state["upsell_count"] = 0
        sandwich_world.tick = 100

        fitness = alex_fitness(sandwich_world, "alex")

        # throughput = 10/100 = 0.1, accuracy = 8/10 = 0.8
        # fitness = 0.1 * 0.8 - 0 + 0 = 0.08
        assert abs(fitness - 0.08) < 0.001

    def test_alex_fitness_tracks_average_customer_wait_ticks(self, sandwich_world) -> None:
        """Test that alex_fitness tracks average_customer_wait_ticks."""
        alex = sandwich_world.agents["alex"]
        alex.internal_state["customers_served"] = 10
        alex.internal_state["total_orders"] = 10
        alex.internal_state["accurate_orders"] = 10
        alex.internal_state["customer_wait_times"] = [20] * 10  # avg 20 ticks
        alex.internal_state["upsell_count"] = 0
        sandwich_world.tick = 100

        fitness = alex_fitness(sandwich_world, "alex")

        # throughput = 10/100 = 0.1, accuracy = 1.0
        # wait_penalty = 20 * 0.05 = 1.0
        # fitness = 0.1 * 1.0 - 1.0 + 0 = -0.9
        assert abs(fitness - (-0.9)) < 0.001

    def test_alex_fitness_counts_upsells(self, sandwich_world) -> None:
        """Test that alex_fitness counts upsells."""
        alex = sandwich_world.agents["alex"]
        alex.internal_state["customers_served"] = 10
        alex.internal_state["total_orders"] = 10
        alex.internal_state["accurate_orders"] = 10
        alex.internal_state["customer_wait_times"] = [0] * 10
        alex.internal_state["upsell_count"] = 5
        sandwich_world.tick = 100

        fitness = alex_fitness(sandwich_world, "alex")

        # throughput = 10/100 = 0.1, accuracy = 1.0
        # upsell_bonus = 5 * 0.1 = 0.5
        # fitness = 0.1 * 1.0 - 0 + 0.5 = 0.6
        assert abs(fitness - 0.6) < 0.001

    def test_alex_fitness_returns_scalar(self, sandwich_world) -> None:
        """Test that alex_fitness returns scalar fitness."""
        alex = sandwich_world.agents["alex"]
        alex.internal_state["customers_served"] = 15
        alex.internal_state["total_orders"] = 15
        alex.internal_state["accurate_orders"] = 12
        alex.internal_state["customer_wait_times"] = [10] * 15
        alex.internal_state["upsell_count"] = 3
        sandwich_world.tick = 200

        fitness = alex_fitness(sandwich_world, "alex")

        assert isinstance(fitness, float)

    def test_alex_fitness_fast_service_higher_fitness(self, sandwich_world) -> None:
        """Test that fast service = higher fitness."""
        # Slow service scenario (long wait times)
        alex = sandwich_world.agents["alex"]
        alex.internal_state["customers_served"] = 10
        alex.internal_state["total_orders"] = 10
        alex.internal_state["accurate_orders"] = 10
        alex.internal_state["customer_wait_times"] = [50] * 10  # slow
        alex.internal_state["upsell_count"] = 0
        sandwich_world.tick = 100

        slow_fitness = alex_fitness(sandwich_world, "alex")

        # Fast service scenario (short wait times)
        alex.internal_state["customer_wait_times"] = [5] * 10  # fast

        fast_fitness = alex_fitness(sandwich_world, "alex")

        assert fast_fitness > slow_fitness

    def test_alex_fitness_long_waits_lower_fitness(self, sandwich_world) -> None:
        """Test that long waits = lower fitness."""
        # Short wait scenario
        alex = sandwich_world.agents["alex"]
        alex.internal_state["customers_served"] = 10
        alex.internal_state["total_orders"] = 10
        alex.internal_state["accurate_orders"] = 10
        alex.internal_state["customer_wait_times"] = [5] * 10
        alex.internal_state["upsell_count"] = 0
        sandwich_world.tick = 100

        short_wait_fitness = alex_fitness(sandwich_world, "alex")

        # Long wait scenario
        alex.internal_state["customer_wait_times"] = [100] * 10

        long_wait_fitness = alex_fitness(sandwich_world, "alex")

        assert short_wait_fitness > long_wait_fitness

    def test_alex_fitness_more_customers_higher_fitness(self, sandwich_world) -> None:
        """Test that more customers = higher fitness."""
        # Few customers scenario
        alex = sandwich_world.agents["alex"]
        alex.internal_state["customers_served"] = 5
        alex.internal_state["total_orders"] = 5
        alex.internal_state["accurate_orders"] = 5
        alex.internal_state["customer_wait_times"] = [0] * 5
        alex.internal_state["upsell_count"] = 0
        sandwich_world.tick = 100

        few_customers_fitness = alex_fitness(sandwich_world, "alex")

        # Many customers scenario
        alex.internal_state["customers_served"] = 20
        alex.internal_state["total_orders"] = 20
        alex.internal_state["accurate_orders"] = 20
        alex.internal_state["customer_wait_times"] = [0] * 20

        many_customers_fitness = alex_fitness(sandwich_world, "alex")

        assert many_customers_fitness > few_customers_fitness

    def test_alex_fitness_handles_zero_ticks(self, sandwich_world) -> None:
        """Test that alex_fitness handles zero ticks gracefully."""
        sandwich_world.tick = 0

        fitness = alex_fitness(sandwich_world, "alex")

        assert fitness == 0.0

    def test_alex_fitness_handles_empty_internal_state(self, sandwich_world) -> None:
        """Test that alex_fitness handles empty internal state."""
        alex = sandwich_world.agents["alex"]
        alex.internal_state.clear()
        sandwich_world.tick = 100

        fitness = alex_fitness(sandwich_world, "alex")

        # All metrics default to 0
        assert fitness == 0.0

    def test_alex_fitness_raises_for_missing_agent(self, sandwich_world) -> None:
        """Test that alex_fitness raises error for missing agent."""
        with pytest.raises(ValueError, match="not found"):
            alex_fitness(sandwich_world, "nonexistent_agent")

    def test_alex_fitness_integration_with_simulation(self, sandwich_world) -> None:
        """Test alex_fitness with actual simulation run."""
        fitness = evaluate_fitness(
            genome={"speed": 0.8, "accuracy": 0.7, "upselling": 0.5},
            world_template=sandwich_world,
            target_agent_id="alex",
            ticks=500,
            fitness_fn=alex_fitness,
            seed=42,
        )

        # Fitness should be a valid number after simulation
        assert isinstance(fitness, float)

    def test_alex_fitness_formula_components(self, sandwich_world) -> None:
        """Test that fitness formula combines all components correctly."""
        alex = sandwich_world.agents["alex"]
        alex.internal_state["customers_served"] = 30
        alex.internal_state["total_orders"] = 30
        alex.internal_state["accurate_orders"] = 27  # 90% accuracy
        alex.internal_state["customer_wait_times"] = [10] * 30  # avg 10 ticks wait
        alex.internal_state["upsell_count"] = 6
        sandwich_world.tick = 500

        fitness = alex_fitness(sandwich_world, "alex")

        # Manual calculation:
        throughput = 30 / 500  # = 0.06
        accuracy = 27 / 30  # = 0.9
        avg_wait = 10
        wait_penalty = avg_wait * 0.05  # = 0.5
        upsell_bonus = 6 * 0.1  # = 0.6

        expected = (throughput * accuracy) - wait_penalty + upsell_bonus
        # = (0.06 * 0.9) - 0.5 + 0.6 = 0.054 - 0.5 + 0.6 = 0.154

        assert abs(fitness - expected) < 0.001


class TestMariaFitness:
    """Test Maria's fitness function per PRD section 9.6."""

    @pytest.fixture
    def sandwich_world(self):
        """Create a sandwich shop world for testing."""
        from loopengine.corpora.sandwich_shop import create_world

        return create_world()

    def test_maria_fitness_computes_shop_total_throughput(self, sandwich_world) -> None:
        """Test that maria_fitness computes shop_total_throughput."""
        maria = sandwich_world.agents["maria"]
        maria.internal_state["supply_cost"] = 0.0
        maria.internal_state["revenue"] = 100.0
        maria.internal_state["stockout_events"] = 0

        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 20
        tom.internal_state["waste_count"] = 0

        sandwich_world.tick = 100

        fitness = maria_fitness(sandwich_world, "maria")

        # throughput = 20/100 = 0.2, margin = 1 - 0/100 = 1.0
        # fitness = 0.2 * 1.0 - 0 - 0 = 0.2
        assert abs(fitness - 0.2) < 0.001

    def test_maria_fitness_tracks_supply_cost_and_revenue(self, sandwich_world) -> None:
        """Test that maria_fitness tracks supply_cost and revenue for margin."""
        maria = sandwich_world.agents["maria"]
        maria.internal_state["supply_cost"] = 30.0
        maria.internal_state["revenue"] = 100.0
        maria.internal_state["stockout_events"] = 0

        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 10
        tom.internal_state["waste_count"] = 0

        sandwich_world.tick = 100

        fitness = maria_fitness(sandwich_world, "maria")

        # throughput = 10/100 = 0.1, margin = 1 - 30/100 = 0.7
        # fitness = 0.1 * 0.7 - 0 - 0 = 0.07
        assert abs(fitness - 0.07) < 0.001

    def test_maria_fitness_counts_stockout_events(self, sandwich_world) -> None:
        """Test that maria_fitness counts stockout_events."""
        maria = sandwich_world.agents["maria"]
        maria.internal_state["supply_cost"] = 0.0
        maria.internal_state["revenue"] = 100.0
        maria.internal_state["stockout_events"] = 3

        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 10
        tom.internal_state["waste_count"] = 0

        sandwich_world.tick = 100

        fitness = maria_fitness(sandwich_world, "maria")

        # throughput = 10/100 = 0.1, margin = 1.0
        # stockout_penalty = 3 * 2.0 = 6.0
        # fitness = 0.1 * 1.0 - 6.0 - 0 = -5.9
        expected = 0.1 * 1.0 - (3 * 2.0) - 0
        assert abs(fitness - expected) < 0.001

    def test_maria_fitness_sums_waste_total(self, sandwich_world) -> None:
        """Test that maria_fitness sums waste_total."""
        maria = sandwich_world.agents["maria"]
        maria.internal_state["supply_cost"] = 0.0
        maria.internal_state["revenue"] = 100.0
        maria.internal_state["stockout_events"] = 0

        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 10
        tom.internal_state["waste_count"] = 4

        sandwich_world.tick = 100

        fitness = maria_fitness(sandwich_world, "maria")

        # throughput = 10/100 = 0.1, margin = 1.0
        # waste_penalty = 4 * 0.5 = 2.0
        # fitness = 0.1 * 1.0 - 0 - 2.0 = -1.9
        expected = 0.1 * 1.0 - 0 - (4 * 0.5)
        assert abs(fitness - expected) < 0.001

    def test_maria_fitness_returns_scalar(self, sandwich_world) -> None:
        """Test that maria_fitness returns scalar fitness."""
        maria = sandwich_world.agents["maria"]
        maria.internal_state["supply_cost"] = 20.0
        maria.internal_state["revenue"] = 80.0
        maria.internal_state["stockout_events"] = 1

        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 15
        tom.internal_state["waste_count"] = 2

        sandwich_world.tick = 150

        fitness = maria_fitness(sandwich_world, "maria")

        assert isinstance(fitness, float)

    def test_maria_fitness_stockouts_heavily_penalized(self, sandwich_world) -> None:
        """Test that stockouts heavily penalize fitness."""
        # Scenario without stockouts
        maria = sandwich_world.agents["maria"]
        maria.internal_state["supply_cost"] = 10.0
        maria.internal_state["revenue"] = 100.0
        maria.internal_state["stockout_events"] = 0

        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 20
        tom.internal_state["waste_count"] = 0

        sandwich_world.tick = 100

        no_stockout_fitness = maria_fitness(sandwich_world, "maria")

        # Scenario with stockouts
        maria.internal_state["stockout_events"] = 5

        with_stockout_fitness = maria_fitness(sandwich_world, "maria")

        # Stockouts should heavily penalize (5 * 2.0 = 10.0 reduction)
        assert no_stockout_fitness - with_stockout_fitness == pytest.approx(10.0)

    def test_maria_fitness_high_throughput_higher_fitness(self, sandwich_world) -> None:
        """Test that high throughput = higher fitness."""
        maria = sandwich_world.agents["maria"]
        maria.internal_state["supply_cost"] = 10.0
        maria.internal_state["revenue"] = 100.0
        maria.internal_state["stockout_events"] = 0

        tom = sandwich_world.agents["tom"]
        tom.internal_state["waste_count"] = 0

        sandwich_world.tick = 100

        # Low throughput
        tom.internal_state["sandwiches_completed"] = 5
        low_throughput_fitness = maria_fitness(sandwich_world, "maria")

        # High throughput
        tom.internal_state["sandwiches_completed"] = 30
        high_throughput_fitness = maria_fitness(sandwich_world, "maria")

        assert high_throughput_fitness > low_throughput_fitness

    def test_maria_fitness_handles_zero_ticks(self, sandwich_world) -> None:
        """Test that maria_fitness handles zero ticks gracefully."""
        sandwich_world.tick = 0

        fitness = maria_fitness(sandwich_world, "maria")

        assert fitness == 0.0

    def test_maria_fitness_handles_zero_revenue(self, sandwich_world) -> None:
        """Test that maria_fitness handles zero revenue (margin = 0)."""
        maria = sandwich_world.agents["maria"]
        maria.internal_state["supply_cost"] = 50.0
        maria.internal_state["revenue"] = 0.0  # No revenue
        maria.internal_state["stockout_events"] = 0

        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 10
        tom.internal_state["waste_count"] = 0

        sandwich_world.tick = 100

        fitness = maria_fitness(sandwich_world, "maria")

        # margin = 0 when revenue = 0, so throughput * margin = 0
        # fitness = 0 - 0 - 0 = 0
        assert fitness == 0.0

    def test_maria_fitness_handles_empty_internal_state(self, sandwich_world) -> None:
        """Test that maria_fitness handles empty internal state."""
        maria = sandwich_world.agents["maria"]
        maria.internal_state.clear()

        tom = sandwich_world.agents["tom"]
        tom.internal_state.clear()

        sandwich_world.tick = 100

        fitness = maria_fitness(sandwich_world, "maria")

        # All metrics default to 0
        assert fitness == 0.0

    def test_maria_fitness_raises_for_missing_agent(self, sandwich_world) -> None:
        """Test that maria_fitness raises error for missing agent."""
        with pytest.raises(ValueError, match="not found"):
            maria_fitness(sandwich_world, "nonexistent_agent")

    def test_maria_fitness_formula_components(self, sandwich_world) -> None:
        """Test that fitness formula combines all components correctly."""
        maria = sandwich_world.agents["maria"]
        maria.internal_state["supply_cost"] = 40.0
        maria.internal_state["revenue"] = 200.0
        maria.internal_state["stockout_events"] = 2

        tom = sandwich_world.agents["tom"]
        tom.internal_state["sandwiches_completed"] = 25
        tom.internal_state["waste_count"] = 6

        sandwich_world.tick = 500

        fitness = maria_fitness(sandwich_world, "maria")

        # Manual calculation:
        throughput = 25 / 500  # = 0.05
        margin = 1 - (40 / 200)  # = 0.8
        stockout_penalty = 2 * 2.0  # = 4.0
        waste_penalty = 6 * 0.5  # = 3.0

        expected = (throughput * margin) - stockout_penalty - waste_penalty
        # = (0.05 * 0.8) - 4.0 - 3.0 = 0.04 - 4.0 - 3.0 = -6.96

        assert abs(fitness - expected) < 0.001

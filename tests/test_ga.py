"""Tests for the genetic algorithm engine (loopengine.engine.ga)."""

from __future__ import annotations

import pytest

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

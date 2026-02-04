"""Tests for flexibility-based input randomization (loopengine.engine.flexibility)."""

from __future__ import annotations

import random

import pytest

from loopengine.engine.flexibility import (
    FlexibilityAwareEvaluator,
    FlexibilityConfig,
    adjust_external_input_variance,
    compute_effective_seed,
    compute_variance_multiplier,
    evaluate_with_flexibility,
    get_role_flexibility,
    inject_perturbation,
)
from loopengine.engine.simulation import tick_world_with_flexibility
from loopengine.model.genome import GenomeSchema, GenomeTrait
from loopengine.model.world import World


@pytest.fixture
def simple_schema() -> GenomeSchema:
    """Create a simple test schema with flexibility score."""
    return GenomeSchema(
        role="test_role",
        traits={
            "speed": GenomeTrait(name="speed", description="Speed trait"),
        },
        flexibility_score=0.5,
    )


@pytest.fixture
def low_flexibility_schema() -> GenomeSchema:
    """Create a schema with low flexibility (predictable inputs)."""
    return GenomeSchema(
        role="worker",
        traits={"speed": GenomeTrait(name="speed", description="Speed")},
        flexibility_score=0.2,
    )


@pytest.fixture
def high_flexibility_schema() -> GenomeSchema:
    """Create a schema with high flexibility (unpredictable inputs)."""
    return GenomeSchema(
        role="manager",
        traits={"adaptability": GenomeTrait(name="adaptability", description="Adaptability")},
        flexibility_score=0.8,
    )


@pytest.fixture
def sandwich_world():
    """Create a sandwich shop world for testing."""
    from loopengine.corpora.sandwich_shop import create_world

    return create_world()


class TestGenomeSchemaFlexibility:
    """Test flexibility_score field in GenomeSchema."""

    def test_schema_has_flexibility_score(self) -> None:
        """Test that GenomeSchema has flexibility_score field."""
        schema = GenomeSchema(role="test", flexibility_score=0.7)
        assert hasattr(schema, "flexibility_score")
        assert schema.flexibility_score == 0.7

    def test_schema_default_flexibility_is_0_5(self) -> None:
        """Test that default flexibility is 0.5 (medium)."""
        schema = GenomeSchema(role="test")
        assert schema.flexibility_score == 0.5

    def test_schema_flexibility_can_be_zero(self) -> None:
        """Test that flexibility can be set to 0 (most predictable)."""
        schema = GenomeSchema(role="test", flexibility_score=0.0)
        assert schema.flexibility_score == 0.0

    def test_schema_flexibility_can_be_one(self) -> None:
        """Test that flexibility can be set to 1 (most unpredictable)."""
        schema = GenomeSchema(role="test", flexibility_score=1.0)
        assert schema.flexibility_score == 1.0


class TestComputeEffectiveSeed:
    """Test compute_effective_seed function."""

    def test_low_flexibility_uses_base_seed(self) -> None:
        """Test that low flexibility always uses base seed."""
        seed = compute_effective_seed(base_seed=42, flexibility=0.1, evaluation_index=0)
        assert seed == 42

        seed = compute_effective_seed(base_seed=42, flexibility=0.1, evaluation_index=5)
        assert seed == 42

        seed = compute_effective_seed(base_seed=42, flexibility=0.3, evaluation_index=10)
        assert seed == 42

    def test_medium_flexibility_uses_indexed_seed(self) -> None:
        """Test that medium flexibility uses base_seed + evaluation_index."""
        seed = compute_effective_seed(base_seed=42, flexibility=0.5, evaluation_index=0)
        assert seed == 42

        seed = compute_effective_seed(base_seed=42, flexibility=0.5, evaluation_index=3)
        assert seed == 45

        seed = compute_effective_seed(base_seed=100, flexibility=0.5, evaluation_index=5)
        assert seed == 105

    def test_high_flexibility_uses_random_seed(self) -> None:
        """Test that high flexibility uses random or None seed."""
        random.seed(123)
        seed1 = compute_effective_seed(base_seed=42, flexibility=0.8, evaluation_index=0)
        # First evaluation with high flexibility should return base + random
        assert seed1 is not None
        assert seed1 != 42

        # Second evaluation with high flexibility returns None
        seed2 = compute_effective_seed(base_seed=42, flexibility=0.8, evaluation_index=1)
        assert seed2 is None

    def test_none_base_seed_returns_none(self) -> None:
        """Test that None base_seed always returns None."""
        assert compute_effective_seed(None, 0.1, 0) is None
        assert compute_effective_seed(None, 0.5, 0) is None
        assert compute_effective_seed(None, 0.9, 0) is None


class TestComputeVarianceMultiplier:
    """Test compute_variance_multiplier function."""

    def test_low_flexibility_returns_low_multiplier(self) -> None:
        """Test that low flexibility returns low variance multiplier."""
        config = FlexibilityConfig()
        mult = compute_variance_multiplier(0.0, config)
        assert mult == config.variance_multiplier_low

    def test_high_flexibility_returns_high_multiplier(self) -> None:
        """Test that high flexibility returns high variance multiplier."""
        config = FlexibilityConfig()
        mult = compute_variance_multiplier(1.0, config)
        assert mult == config.variance_multiplier_high

    def test_medium_flexibility_returns_interpolated(self) -> None:
        """Test that medium flexibility returns interpolated value."""
        config = FlexibilityConfig(variance_multiplier_low=0.2, variance_multiplier_high=2.0)
        mult = compute_variance_multiplier(0.5, config)
        # Linear interpolation: 0.2 + (2.0 - 0.2) * 0.5 = 0.2 + 0.9 = 1.1
        assert abs(mult - 1.1) < 0.001

    def test_variance_multiplier_scales_linearly(self) -> None:
        """Test that multiplier scales linearly with flexibility."""
        config = FlexibilityConfig(variance_multiplier_low=0.0, variance_multiplier_high=1.0)

        for flex in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mult = compute_variance_multiplier(flex, config)
            assert abs(mult - flex) < 0.001


class TestAdjustExternalInputVariance:
    """Test adjust_external_input_variance function."""

    def test_adjusts_variance_based_on_flexibility(self, sandwich_world) -> None:
        """Test that external input variance is adjusted."""
        # Get original variance
        original_variance = sandwich_world.external_inputs[0].variance

        # Adjust for low flexibility
        adjust_external_input_variance(sandwich_world, 0.2)

        # Should have reduced variance
        adjusted_variance = sandwich_world.external_inputs[0].variance
        assert adjusted_variance < original_variance or original_variance == 0

    def test_low_flexibility_reduces_variance(self, sandwich_world) -> None:
        """Test that low flexibility reduces variance."""
        # Set a known variance
        sandwich_world.external_inputs[0].variance = 0.5

        config = FlexibilityConfig(variance_multiplier_low=0.1, variance_multiplier_high=2.0)
        adjust_external_input_variance(sandwich_world, 0.0, config)

        # Variance should be reduced (0.5 * 0.1 = 0.05)
        assert abs(sandwich_world.external_inputs[0].variance - 0.05) < 0.001

    def test_high_flexibility_increases_variance(self, sandwich_world) -> None:
        """Test that high flexibility increases variance."""
        # Set a known variance
        sandwich_world.external_inputs[0].variance = 0.5

        config = FlexibilityConfig(variance_multiplier_low=0.1, variance_multiplier_high=2.0)
        adjust_external_input_variance(sandwich_world, 1.0, config)

        # Variance should be increased (0.5 * 2.0 = 1.0)
        assert abs(sandwich_world.external_inputs[0].variance - 1.0) < 0.001


class TestInjectPerturbation:
    """Test inject_perturbation function."""

    def test_zero_flexibility_no_perturbation(self, sandwich_world) -> None:
        """Test that zero flexibility never injects perturbation."""
        random.seed(42)
        # Run many times - should never inject
        perturbations = sum(inject_perturbation(sandwich_world, 0.0) for _ in range(100))
        assert perturbations == 0

    def test_high_flexibility_sometimes_injects(self, sandwich_world) -> None:
        """Test that high flexibility sometimes injects perturbation."""
        random.seed(42)
        config = FlexibilityConfig(perturbation_base_rate=0.5)  # 50% chance at flex=1.0

        perturbations = sum(inject_perturbation(sandwich_world, 1.0, config) for _ in range(100))

        # With 50% rate, expect around 50 perturbations
        assert 20 < perturbations < 80

    def test_perturbation_probability_scales_with_flexibility(self, sandwich_world) -> None:
        """Test that perturbation probability scales with flexibility."""
        config = FlexibilityConfig(perturbation_base_rate=0.1)

        # Count perturbations at different flexibility levels
        random.seed(123)
        low_perturbations = sum(
            inject_perturbation(sandwich_world, 0.2, config) for _ in range(1000)
        )

        random.seed(123)
        high_perturbations = sum(
            inject_perturbation(sandwich_world, 0.8, config) for _ in range(1000)
        )

        # Higher flexibility should have more perturbations
        assert high_perturbations > low_perturbations


class TestFlexibilityAwareEvaluator:
    """Test FlexibilityAwareEvaluator class."""

    def test_evaluator_requires_target_agent_id(self, sandwich_world) -> None:
        """Test that evaluator requires target_agent_id."""
        evaluator = FlexibilityAwareEvaluator(fitness_fn=lambda w, a: 1.0)

        with pytest.raises(ValueError, match="target_agent_id"):
            evaluator.evaluate({"speed": 0.5}, sandwich_world)

    def test_evaluator_requires_fitness_fn(self, sandwich_world) -> None:
        """Test that evaluator requires fitness_fn."""
        evaluator = FlexibilityAwareEvaluator(target_agent_id="tom")

        with pytest.raises(ValueError, match="fitness_fn"):
            evaluator.evaluate({"speed": 0.5}, sandwich_world)

    def test_evaluator_raises_for_missing_agent(self, sandwich_world) -> None:
        """Test that evaluator raises error for missing agent."""
        evaluator = FlexibilityAwareEvaluator(
            target_agent_id="nonexistent",
            fitness_fn=lambda w, a: 1.0,
        )

        with pytest.raises(ValueError, match="not found"):
            evaluator.evaluate({"speed": 0.5}, sandwich_world)

    def test_low_flexibility_single_run(self, sandwich_world) -> None:
        """Test that low flexibility uses single evaluation run."""
        run_count = 0

        def counting_fitness(world, agent_id) -> float:
            nonlocal run_count
            run_count += 1
            return 1.0

        evaluator = FlexibilityAwareEvaluator(
            target_agent_id="tom",
            fitness_fn=counting_fitness,
            flexibility=0.3,  # Low flexibility
            ticks=10,
        )

        evaluator.evaluate({"speed": 0.5}, sandwich_world)
        assert run_count == 1

    def test_high_flexibility_multiple_runs(self, sandwich_world) -> None:
        """Test that high flexibility uses multiple evaluation runs."""
        run_count = 0

        def counting_fitness(world, agent_id) -> float:
            nonlocal run_count
            run_count += 1
            return 1.0

        config = FlexibilityConfig(evaluation_runs=3)
        evaluator = FlexibilityAwareEvaluator(
            target_agent_id="tom",
            fitness_fn=counting_fitness,
            flexibility=0.8,  # High flexibility
            ticks=10,
            config=config,
        )

        evaluator.evaluate({"speed": 0.5}, sandwich_world)
        assert run_count == 3

    def test_evaluator_returns_average_fitness(self, sandwich_world) -> None:
        """Test that high flexibility evaluator returns average fitness."""
        call_count = 0

        def varying_fitness(world, agent_id) -> float:
            nonlocal call_count
            call_count += 1
            # Return 1.0, 2.0, 3.0 on successive calls
            return float(call_count)

        config = FlexibilityConfig(evaluation_runs=3)
        evaluator = FlexibilityAwareEvaluator(
            target_agent_id="tom",
            fitness_fn=varying_fitness,
            flexibility=0.8,
            ticks=10,
            config=config,
        )

        fitness = evaluator.evaluate({"speed": 0.5}, sandwich_world)
        # Average of 1, 2, 3 = 2.0
        assert abs(fitness - 2.0) < 0.001

    def test_evaluator_applies_genome(self, sandwich_world) -> None:
        """Test that evaluator applies genome to target agent."""
        test_genome = {"speed": 0.99, "consistency": 0.01}

        def genome_check_fitness(world, agent_id) -> float:
            return world.agents[agent_id].genome.get("speed", 0)

        evaluator = FlexibilityAwareEvaluator(
            target_agent_id="tom",
            fitness_fn=genome_check_fitness,
            flexibility=0.5,
            ticks=1,
        )

        fitness = evaluator.evaluate(test_genome, sandwich_world)
        assert abs(fitness - 0.99) < 0.001


class TestEvaluateWithFlexibility:
    """Test evaluate_with_flexibility convenience function."""

    def test_evaluate_with_low_flexibility(self, sandwich_world) -> None:
        """Test evaluation with low flexibility."""
        fitness = evaluate_with_flexibility(
            genome={"speed": 0.7},
            world_template=sandwich_world,
            target_agent_id="tom",
            flexibility=0.2,
            ticks=100,
            fitness_fn=lambda w, a: float(w.tick),
        )

        assert fitness == 100.0

    def test_evaluate_with_high_flexibility(self, sandwich_world) -> None:
        """Test evaluation with high flexibility."""
        config = FlexibilityConfig(evaluation_runs=3)

        fitness = evaluate_with_flexibility(
            genome={"speed": 0.7},
            world_template=sandwich_world,
            target_agent_id="tom",
            flexibility=0.8,
            ticks=100,
            fitness_fn=lambda w, a: float(w.tick),
            config=config,
        )

        # Should still return 100 (each run goes 100 ticks)
        assert fitness == 100.0

    def test_evaluate_deterministic_with_low_flexibility(self, sandwich_world) -> None:
        """Test that low flexibility evaluation is deterministic."""

        def varying_fitness(world, agent_id) -> float:
            return sum(a.internal_state.get("served_count", 0) for a in world.agents.values())

        fitness1 = evaluate_with_flexibility(
            genome={"speed": 0.7},
            world_template=sandwich_world,
            target_agent_id="tom",
            flexibility=0.1,  # Low flexibility = consistent seed
            ticks=200,
            fitness_fn=varying_fitness,
            base_seed=42,
        )

        fitness2 = evaluate_with_flexibility(
            genome={"speed": 0.7},
            world_template=sandwich_world,
            target_agent_id="tom",
            flexibility=0.1,
            ticks=200,
            fitness_fn=varying_fitness,
            base_seed=42,
        )

        assert fitness1 == fitness2


class TestGetRoleFlexibility:
    """Test get_role_flexibility function."""

    def test_returns_schema_flexibility(self) -> None:
        """Test that it returns flexibility from schema."""
        world = World(
            schemas={
                "worker": GenomeSchema(role="worker", flexibility_score=0.3),
                "manager": GenomeSchema(role="manager", flexibility_score=0.8),
            }
        )

        assert get_role_flexibility(world, "worker") == 0.3
        assert get_role_flexibility(world, "manager") == 0.8

    def test_returns_default_for_unknown_role(self) -> None:
        """Test that it returns 0.5 for unknown roles."""
        world = World()
        assert get_role_flexibility(world, "unknown") == 0.5


class TestTickWorldWithFlexibility:
    """Test tick_world_with_flexibility function."""

    def test_tick_advances_world(self, sandwich_world) -> None:
        """Test that tick advances the world normally."""
        initial_tick = sandwich_world.tick
        tick_world_with_flexibility(sandwich_world)
        assert sandwich_world.tick == initial_tick + 1

    def test_tick_with_custom_flexibility(self, sandwich_world) -> None:
        """Test tick with custom flexibility map."""
        random.seed(42)
        flexibility_map = {"sandwich_maker": 0.3, "cashier": 0.4, "owner": 0.8}

        # Run several ticks
        for _ in range(100):
            tick_world_with_flexibility(sandwich_world, flexibility_map)

        # World should have advanced
        assert sandwich_world.tick == 100

    def test_high_flexibility_injects_perturbations(self, sandwich_world) -> None:
        """Test that high flexibility leads to perturbations over time."""
        random.seed(42)
        flexibility_map = {"sandwich_maker": 1.0, "cashier": 1.0, "owner": 1.0}

        perturbation_count = 0
        for _ in range(1000):
            if tick_world_with_flexibility(sandwich_world, flexibility_map):
                perturbation_count += 1

        # Should have some perturbations with high flexibility
        assert perturbation_count > 0


class TestFlexibilityIntegration:
    """Integration tests for flexibility system."""

    def test_discovered_schema_has_flexibility(self) -> None:
        """Test that discovered schemas include flexibility_score."""
        from loopengine.discovery.discoverer import DiscoveredRole, DiscoveryResult

        # Simulate what the discoverer produces
        schema = GenomeSchema(
            role="sandwich_maker",
            traits={"speed": GenomeTrait(name="speed", description="Speed")},
            flexibility_score=0.35,
        )

        result = DiscoveryResult(
            roles={
                "sandwich_maker": DiscoveredRole(
                    schema=schema,
                    flexibility_score=0.35,
                )
            }
        )

        assert result.roles["sandwich_maker"].schema.flexibility_score == 0.35

    def test_flexibility_affects_evolved_genome_robustness(self, sandwich_world) -> None:
        """Test that high flexibility evaluation selects for robustness.

        This is a conceptual test - with varied inputs, the best genome
        should be one that performs consistently well across conditions.
        """

        # Simple fitness that varies based on random state
        def variable_fitness(world, agent_id) -> float:
            base = sum(a.internal_state.get("served_count", 0) for a in world.agents.values())
            return base

        # Low flexibility: deterministic
        random.seed(42)
        low_flex_fitness = evaluate_with_flexibility(
            genome={"speed": 0.7},
            world_template=sandwich_world,
            target_agent_id="tom",
            flexibility=0.2,
            ticks=100,
            fitness_fn=variable_fitness,
        )

        # High flexibility: averaged over multiple runs
        random.seed(42)
        high_flex_fitness = evaluate_with_flexibility(
            genome={"speed": 0.7},
            world_template=sandwich_world,
            target_agent_id="tom",
            flexibility=0.8,
            ticks=100,
            fitness_fn=variable_fitness,
            config=FlexibilityConfig(evaluation_runs=3),
        )

        # Both should produce valid fitness values
        assert isinstance(low_flex_fitness, float)
        assert isinstance(high_flex_fitness, float)

    def test_world_schemas_used_for_flexibility(self, sandwich_world) -> None:
        """Test that world schemas provide flexibility for roles."""
        # Add schemas to world
        sandwich_world.schemas = {
            "sandwich_maker": GenomeSchema(
                role="sandwich_maker",
                traits={"speed": GenomeTrait(name="speed", description="Speed")},
                flexibility_score=0.35,
            ),
            "cashier": GenomeSchema(
                role="cashier",
                traits={"accuracy": GenomeTrait(name="accuracy", description="Accuracy")},
                flexibility_score=0.45,
            ),
        }

        # Get flexibility from world
        assert get_role_flexibility(sandwich_world, "sandwich_maker") == 0.35
        assert get_role_flexibility(sandwich_world, "cashier") == 0.45
        assert get_role_flexibility(sandwich_world, "unknown") == 0.5  # default

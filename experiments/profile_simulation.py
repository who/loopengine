"""Profile tick_world() to identify performance bottlenecks."""

import cProfile
import pstats
import time
from io import StringIO

from loopengine.corpora.sandwich_shop import create_world
from loopengine.engine.simulation import MAX_PARTICLES, tick_world
from loopengine.model.particle import Particle
from loopengine.model.world import World


def create_test_world() -> World:
    """Create a test world with sandwich shop corpus."""
    return create_world()


def measure_tick_rate(world: World, num_ticks: int) -> tuple[float, int]:
    """Measure ticks per second and particle count."""
    start_time = time.perf_counter()
    max_particles = 0

    for _ in range(num_ticks):
        tick_world(world)
        max_particles = max(max_particles, len(world.particles))

    elapsed = time.perf_counter() - start_time
    ticks_per_sec = num_ticks / elapsed if elapsed > 0 else 0
    return ticks_per_sec, max_particles


def profile_tick_world(world: World, num_ticks: int) -> str:
    """Profile tick_world and return profiling results."""
    profiler = cProfile.Profile()

    profiler.enable()
    for _ in range(num_ticks):
        tick_world(world)
    profiler.disable()

    # Get stats sorted by cumulative time
    stats_stream = StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)  # Top 30 functions

    return stats_stream.getvalue()


def stress_test_high_particles(world: World) -> tuple[float, int, int]:
    """Stress test with high external input rates."""
    # Increase external input rate to generate more particles
    for ext_input in world.external_inputs:
        ext_input.rate *= 10  # 10x the normal rate for stress test

    # Run for 500 ticks
    start_time = time.perf_counter()
    max_particles = 0
    total_particles_created = 0

    for _ in range(500):
        pre_count = len(world.particles)
        tick_world(world)
        post_count = len(world.particles)
        max_particles = max(max_particles, post_count)
        if post_count > pre_count:
            total_particles_created += post_count - pre_count

    elapsed = time.perf_counter() - start_time
    ticks_per_sec = 500 / elapsed if elapsed > 0 else 0

    return ticks_per_sec, max_particles, total_particles_created


def test_with_50_particles(world: World) -> tuple[float, int]:
    """Test performance with exactly 50+ concurrent particles.

    Injects particles directly to reach target count.
    """
    # Inject 50 particles directly
    for i in range(50):
        particle = Particle(
            id=f"test_particle_{i}",
            particle_type="test",
            payload={},
            source_id="alex",
            dest_id="tom",
            link_id="alex_to_tom",
            progress=0.1 + (i * 0.01),  # Spread across the link
            speed=0.05,
            alive=True,
        )
        world.particles[particle.id] = particle

    # Run for 1000 ticks and track particles
    start_time = time.perf_counter()
    max_particles = 50

    for _ in range(1000):
        tick_world(world)
        max_particles = max(max_particles, len(world.particles))

    elapsed = time.perf_counter() - start_time
    ticks_per_sec = 1000 / elapsed if elapsed > 0 else 0

    return ticks_per_sec, max_particles


def test_memory_stability(world: World, duration_ticks: int = 5000) -> tuple[float, int, int]:
    """Test for memory stability over long runs."""
    import gc

    # Force garbage collection before starting
    gc.collect()

    start_time = time.perf_counter()
    max_particles = 0
    particles_at_start = len(world.particles)

    for tick in range(duration_ticks):
        tick_world(world)
        current_particles = len(world.particles)
        max_particles = max(max_particles, current_particles)

        # Check for memory leaks every 1000 ticks
        if tick % 1000 == 0:
            gc.collect()

    elapsed = time.perf_counter() - start_time
    ticks_per_sec = duration_ticks / elapsed if elapsed > 0 else 0
    particles_at_end = len(world.particles)

    return ticks_per_sec, particles_at_start, particles_at_end


def main():
    print("=" * 60)
    print("Performance Profiling: tick_world()")
    print("=" * 60)

    # Create test world
    world = create_test_world()
    print(f"\nWorld setup: {len(world.agents)} agents, {len(world.links)} links")
    print(f"Max particle limit: {MAX_PARTICLES}")

    # Warm-up run
    print("\nWarm-up run (100 ticks)...")
    measure_tick_rate(world, 100)

    # Reset world
    world = create_test_world()

    # Normal rate measurement (1000 ticks)
    print("\n--- Normal Rate Measurement (1000 ticks) ---")
    ticks_per_sec, max_particles = measure_tick_rate(world, 1000)
    print(f"Tick rate: {ticks_per_sec:.1f} ticks/sec")
    print(f"Max particles: {max_particles}")

    # Test with 50 particles (target from acceptance criteria)
    print("\n--- Test with 50 Particles (1000 ticks) ---")
    world = create_test_world()
    ticks_per_sec_50, max_particles_50 = test_with_50_particles(world)
    print(f"Tick rate with 50+ particles: {ticks_per_sec_50:.1f} ticks/sec")
    print(f"Max concurrent particles: {max_particles_50}")

    # Stress test with high particle count
    print("\n--- Stress Test (10x input rate, 500 ticks) ---")
    world = create_test_world()
    ticks_per_sec, max_particles, total_created = stress_test_high_particles(world)
    print(f"Tick rate under stress: {ticks_per_sec:.1f} ticks/sec")
    print(f"Max concurrent particles: {max_particles}")
    print(f"Total particles created: {total_created}")

    # Memory stability test
    print("\n--- Memory Stability Test (5000 ticks) ---")
    world = create_test_world()
    ticks_per_sec_long, start_particles, end_particles = test_memory_stability(world, 5000)
    print(f"Tick rate over long run: {ticks_per_sec_long:.1f} ticks/sec")
    print(f"Particles at start: {start_particles}")
    print(f"Particles at end: {end_particles}")

    # Profile detailed breakdown
    print("\n--- Profiling Breakdown (1000 ticks) ---")
    world = create_test_world()
    profile_results = profile_tick_world(world, 1000)
    print(profile_results)

    # Final assessment
    print("\n" + "=" * 60)
    print("Performance Assessment")
    print("=" * 60)

    target_ticks = 60
    all_passed = True

    # Test 1: 60+ ticks/sec with 50 particles
    if ticks_per_sec_50 >= target_ticks:
        print(f"✓ PASS: {ticks_per_sec_50:.0f} ticks/sec with 50 particles (target: 60)")
    else:
        print(f"✗ FAIL: {ticks_per_sec_50:.0f} ticks/sec (target: 60)")
        all_passed = False

    # Test 2: Particle limit enforced
    if max_particles <= MAX_PARTICLES:
        print(f"✓ PASS: Particle count capped at {max_particles} (limit: {MAX_PARTICLES})")
    else:
        print(f"✗ FAIL: Particle count {max_particles} exceeded limit {MAX_PARTICLES}")
        all_passed = False

    # Test 3: No memory leak (particles don't grow unbounded)
    if end_particles <= MAX_PARTICLES:
        print(f"✓ PASS: No memory leak detected (particles: {end_particles})")
    else:
        print(f"✗ FAIL: Memory leak detected (particles: {end_particles})")
        all_passed = False

    print()
    if all_passed:
        print("All performance tests PASSED!")
    else:
        print("Some performance tests FAILED.")


if __name__ == "__main__":
    main()

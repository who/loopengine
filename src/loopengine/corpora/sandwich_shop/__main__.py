"""Console runner for the Sandwich Shop simulation.

Usage:
    uv run python -m loopengine.corpora.sandwich_shop
    uv run python -m loopengine.corpora.sandwich_shop --ticks 500
"""

from __future__ import annotations

import argparse
import sys

from loopengine.corpora.sandwich_shop import create_world
from loopengine.engine import tick_world
from loopengine.logging_config import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Sandwich Shop simulation headlessly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=1000,
        help="Number of simulation ticks to run",
    )
    parser.add_argument(
        "--summary-interval",
        type=int,
        default=100,
        help="Print state summary every N ticks",
    )
    return parser.parse_args(argv)


def print_state_summary(world) -> None:
    """Print a state summary for the current tick."""
    # Collect agent phases
    agent_phases = {agent.id: agent.loop_phase.value for agent in world.agents.values()}

    # Count active particles
    particle_count = len(world.particles)

    # Calculate queue depths (input buffer sizes)
    queue_depths = {agent.id: len(agent.input_buffer) for agent in world.agents.values()}

    # Format output
    print(f"Tick {world.tick:5d} | Particles: {particle_count:3d} | ", end="")
    print("Phases: ", end="")
    for agent_id, phase in sorted(agent_phases.items()):
        print(f"{agent_id}={phase[0].upper()} ", end="")
    print("| Queues: ", end="")
    for agent_id, depth in sorted(queue_depths.items()):
        print(f"{agent_id}={depth} ", end="")
    print()


def main(argv: list[str] | None = None) -> int:
    """Run the sandwich shop simulation."""
    args = parse_args(argv)

    # Configure logging before any other operations
    configure_logging()

    print(f"Starting Sandwich Shop simulation for {args.ticks} ticks...")
    print("=" * 70)

    world = create_world()

    # Print initial state
    print_state_summary(world)

    # Run simulation
    for _ in range(args.ticks):
        tick_world(world)

        # Print summary at intervals
        if world.tick % args.summary_interval == 0:
            print_state_summary(world)

    print("=" * 70)
    print(f"Simulation complete. Final tick: {world.tick}")

    # Print final statistics
    total_served = 0
    for agent in world.agents.values():
        served = agent.internal_state.get("served_count", 0)
        total_served += served

    print(f"Customers served: {total_served}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

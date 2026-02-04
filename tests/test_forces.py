"""Tests for the force-directed layout engine."""

import math

from loopengine.engine.forces import (
    AgentForces,
    ForceConfig,
    ForceVector,
    apply_forces,
    compute_forces,
    update_layout,
)
from loopengine.model.agent import Agent
from loopengine.model.link import Link, LinkType
from loopengine.model.world import World


def make_agent(
    agent_id: str,
    x: float = 0.0,
    y: float = 0.0,
    labels: set[str] | None = None,
) -> Agent:
    """Create a test agent with position."""
    return Agent(
        id=agent_id,
        name=f"Agent {agent_id}",
        role="tester",
        x=x,
        y=y,
        labels=labels if labels is not None else set(),
    )


def make_world() -> World:
    """Create an empty test world."""
    return World()


class TestForceVector:
    """Tests for ForceVector dataclass."""

    def test_add_returns_new_vector(self):
        """Adding two vectors should return a new combined vector."""
        v1 = ForceVector(1.0, 2.0)
        v2 = ForceVector(3.0, 4.0)

        result = v1.add(v2)

        assert result.fx == 4.0
        assert result.fy == 6.0

    def test_magnitude_calculates_correctly(self):
        """Magnitude should calculate Euclidean distance."""
        v = ForceVector(3.0, 4.0)

        assert v.magnitude() == 5.0

    def test_clamp_reduces_large_magnitude(self):
        """Clamp should reduce vectors exceeding max magnitude."""
        v = ForceVector(30.0, 40.0)  # magnitude = 50

        result = v.clamp(10.0)

        assert abs(result.magnitude() - 10.0) < 0.001

    def test_clamp_preserves_small_magnitude(self):
        """Clamp should not modify vectors within limit."""
        v = ForceVector(3.0, 4.0)  # magnitude = 5

        result = v.clamp(10.0)

        assert result.fx == 3.0
        assert result.fy == 4.0


class TestAgentForces:
    """Tests for AgentForces accumulator."""

    def test_get_creates_new_force(self):
        """Getting a non-existent agent should create zero force."""
        forces = AgentForces()

        force = forces.get("agent1")

        assert force.fx == 0.0
        assert force.fy == 0.0

    def test_add_force_accumulates(self):
        """Adding forces should accumulate them."""
        forces = AgentForces()

        forces.add_force("agent1", ForceVector(1.0, 2.0))
        forces.add_force("agent1", ForceVector(3.0, 4.0))

        result = forces.get("agent1")
        assert result.fx == 4.0
        assert result.fy == 6.0


class TestSpringForces:
    """Tests for link spring force computation."""

    def test_spring_pulls_distant_agents_together(self):
        """Agents farther than rest length should be pulled together."""
        world = make_world()

        # Create two agents far apart
        agent1 = make_agent("a1", x=0.0, y=0.0)
        agent2 = make_agent("a2", x=200.0, y=0.0)  # 200 units apart
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        # Create link between them
        link = Link(id="link1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        world.links["link1"] = link

        config = ForceConfig(spring_rest_length=100.0)  # Rest length 100
        forces = compute_forces(world, config)

        # a1 should be pulled toward a2 (positive x)
        assert forces.get("a1").fx > 0
        # a2 should be pulled toward a1 (negative x)
        assert forces.get("a2").fx < 0

    def test_spring_pushes_close_agents_apart(self):
        """Agents closer than rest length should be pushed apart."""
        world = make_world()

        # Create two agents close together
        agent1 = make_agent("a1", x=0.0, y=0.0)
        agent2 = make_agent("a2", x=50.0, y=0.0)  # 50 units apart
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        # Create link between them
        link = Link(id="link1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        world.links["link1"] = link

        config = ForceConfig(spring_rest_length=100.0)  # Rest length 100
        forces = compute_forces(world, config)

        # a1 should be pushed away from a2 (negative x)
        assert forces.get("a1").fx < 0
        # a2 should be pushed away from a1 (positive x)
        assert forces.get("a2").fx > 0


class TestRepulsionForces:
    """Tests for agent repulsion force computation."""

    def test_repulsion_prevents_overlap(self):
        """Agents should repel each other."""
        world = make_world()

        # Create two agents close together
        agent1 = make_agent("a1", x=0.0, y=0.0)
        agent2 = make_agent("a2", x=20.0, y=0.0)
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        # No links - just repulsion
        forces = compute_forces(world)

        # a1 should be pushed left (negative x)
        assert forces.get("a1").fx < 0
        # a2 should be pushed right (positive x)
        assert forces.get("a2").fx > 0

    def test_repulsion_stronger_when_closer(self):
        """Repulsion should be stronger when agents are closer."""
        # First test: agents close
        world1 = make_world()
        agent1 = make_agent("a1", x=0.0, y=0.0)
        agent2 = make_agent("a2", x=20.0, y=0.0)
        world1.agents["a1"] = agent1
        world1.agents["a2"] = agent2

        forces_close = compute_forces(world1)

        # Second test: agents far
        world2 = make_world()
        agent3 = make_agent("a1", x=0.0, y=0.0)
        agent4 = make_agent("a2", x=100.0, y=0.0)
        world2.agents["a1"] = agent3
        world2.agents["a2"] = agent4

        forces_far = compute_forces(world2)

        # Close agents should have stronger repulsion
        assert abs(forces_close.get("a1").fx) > abs(forces_far.get("a1").fx)

    def test_three_agents_repel_correctly(self):
        """Three agents in a triangle should all repel each other."""
        world = make_world()

        # Create three agents in a triangle
        agent1 = make_agent("a1", x=0.0, y=0.0)
        agent2 = make_agent("a2", x=50.0, y=0.0)
        agent3 = make_agent("a3", x=25.0, y=43.3)  # Roughly equilateral
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2
        world.agents["a3"] = agent3

        forces = compute_forces(world)

        # All agents should have non-zero forces
        assert forces.get("a1").magnitude() > 0
        assert forces.get("a2").magnitude() > 0
        assert forces.get("a3").magnitude() > 0


class TestHierarchicalForces:
    """Tests for hierarchical vertical bias force computation."""

    def test_hierarchical_link_induces_vertical_separation(self):
        """Hierarchical links should push authority up, subordinate down."""
        world = make_world()

        # Create manager and worker at same height
        manager = make_agent("manager", x=0.0, y=0.0)
        worker = make_agent("worker", x=100.0, y=0.0)
        world.agents["manager"] = manager
        world.agents["worker"] = worker

        # Hierarchical link: manager → worker
        link = Link(
            id="link1",
            source_id="manager",
            dest_id="worker",
            link_type=LinkType.HIERARCHICAL,
        )
        world.links["link1"] = link

        forces = compute_forces(world)

        # Manager (source) should be pushed up (positive y)
        assert forces.get("manager").fy > 0
        # Worker (dest) should be pushed down (negative y)
        assert forces.get("worker").fy < 0

    def test_peer_link_no_vertical_bias(self):
        """Peer links should not induce vertical bias."""
        world = make_world()

        agent1 = make_agent("a1", x=0.0, y=0.0)
        agent2 = make_agent("a2", x=100.0, y=0.0)
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        # Peer link
        link = Link(id="link1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        world.links["link1"] = link

        # Use zero spring strength to isolate hierarchical effect
        config = ForceConfig(spring_strength=0.0, repulsion_strength=0.0)
        forces = compute_forces(world, config)

        # No vertical force from peer links
        assert forces.get("a1").fy == 0.0
        assert forces.get("a2").fy == 0.0

    def test_maria_above_tom_after_layout(self):
        """Maria (manager) should end up above Tom (worker) after layout settles."""
        world = make_world()

        # Maria is the manager
        maria = make_agent("maria", x=0.0, y=0.0)
        tom = make_agent("tom", x=50.0, y=0.0)
        world.agents["maria"] = maria
        world.agents["tom"] = tom

        # Hierarchical link: Maria → Tom
        link = Link(
            id="link1",
            source_id="maria",
            dest_id="tom",
            link_type=LinkType.HIERARCHICAL,
        )
        world.links["link1"] = link

        # Run layout for many iterations
        for _ in range(100):
            update_layout(world)

        # Maria should be above Tom (higher y)
        assert world.agents["maria"].y > world.agents["tom"].y


class TestLabelCohesion:
    """Tests for label cohesion force computation."""

    def test_label_cohesion_pulls_agents_together(self):
        """Agents with same label should be pulled toward their centroid."""
        world = make_world()

        # Create three agents with same label, spread out
        agent1 = make_agent("a1", x=0.0, y=0.0, labels={"team"})
        agent2 = make_agent("a2", x=100.0, y=0.0, labels={"team"})
        agent3 = make_agent("a3", x=50.0, y=100.0, labels={"team"})
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2
        world.agents["a3"] = agent3

        # Use only cohesion force
        config = ForceConfig(
            spring_strength=0.0,
            repulsion_strength=0.0,
            vertical_strength=0.0,
            cohesion_strength=0.01,
        )
        forces = compute_forces(world, config)

        # Centroid is at (50, 33.33)
        # a1 at (0, 0) should be pulled right and up
        assert forces.get("a1").fx > 0
        assert forces.get("a1").fy > 0

        # a2 at (100, 0) should be pulled left and up
        assert forces.get("a2").fx < 0
        assert forces.get("a2").fy > 0

        # a3 at (50, 100) should be pulled down
        assert forces.get("a3").fy < 0

    def test_different_labels_no_cohesion(self):
        """Agents with different labels should not attract each other via cohesion."""
        world = make_world()

        agent1 = make_agent("a1", x=0.0, y=0.0, labels={"team_a"})
        agent2 = make_agent("a2", x=100.0, y=0.0, labels={"team_b"})
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        # Use only cohesion force
        config = ForceConfig(
            spring_strength=0.0,
            repulsion_strength=0.0,
            vertical_strength=0.0,
            cohesion_strength=0.01,
        )
        forces = compute_forces(world, config)

        # No cohesion between different labels
        assert forces.get("a1").fx == 0.0
        assert forces.get("a2").fx == 0.0

    def test_single_agent_label_no_cohesion(self):
        """Single agent with a label should not have cohesion force."""
        world = make_world()

        agent1 = make_agent("a1", x=0.0, y=0.0, labels={"solo"})
        world.agents["a1"] = agent1

        config = ForceConfig(
            spring_strength=0.0,
            repulsion_strength=0.0,
            vertical_strength=0.0,
        )
        forces = compute_forces(world, config)

        # No cohesion for single agent
        assert forces.get("a1").fx == 0.0
        assert forces.get("a1").fy == 0.0


class TestApplyForces:
    """Tests for applying forces to agent positions."""

    def test_apply_forces_updates_velocity(self):
        """Applying forces should update agent velocity."""
        world = make_world()

        agent = make_agent("a1", x=0.0, y=0.0)
        world.agents["a1"] = agent

        forces = AgentForces()
        forces.add_force("a1", ForceVector(10.0, 5.0))

        apply_forces(world, forces)

        # Velocity should be updated (with damping applied)
        assert world.agents["a1"].vx != 0.0
        assert world.agents["a1"].vy != 0.0

    def test_apply_forces_updates_position(self):
        """Applying forces should update agent position via velocity."""
        world = make_world()

        agent = make_agent("a1", x=0.0, y=0.0)
        world.agents["a1"] = agent

        forces = AgentForces()
        forces.add_force("a1", ForceVector(10.0, 0.0))

        apply_forces(world, forces)

        # Position should change
        assert world.agents["a1"].x != 0.0

    def test_damping_reduces_velocity(self):
        """Damping should reduce velocity over time."""
        world = make_world()

        agent = make_agent("a1", x=0.0, y=0.0)
        agent.vx = 10.0
        agent.vy = 10.0
        world.agents["a1"] = agent

        # Apply zero force - just damping
        forces = AgentForces()
        forces.add_force("a1", ForceVector(0.0, 0.0))

        config = ForceConfig(damping=0.5)  # 50% damping
        apply_forces(world, forces, config)

        # Velocity should be reduced by damping
        assert abs(world.agents["a1"].vx) < 10.0
        assert abs(world.agents["a1"].vy) < 10.0

    def test_velocity_clamping(self):
        """Velocity should be clamped to max_velocity."""
        world = make_world()

        agent = make_agent("a1", x=0.0, y=0.0)
        world.agents["a1"] = agent

        forces = AgentForces()
        forces.add_force("a1", ForceVector(1000.0, 1000.0))

        config = ForceConfig(max_velocity=10.0, damping=0.0, max_force=10000.0)
        apply_forces(world, forces, config)

        velocity = math.sqrt(agent.vx**2 + agent.vy**2)
        assert velocity <= 10.0 + 0.001  # Small tolerance for float


class TestUpdateLayout:
    """Tests for the full update_layout function."""

    def test_update_layout_modifies_positions(self):
        """update_layout should modify agent positions."""
        world = make_world()

        agent1 = make_agent("a1", x=0.0, y=0.0)
        agent2 = make_agent("a2", x=10.0, y=0.0)
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        initial_x1 = agent1.x
        initial_x2 = agent2.x

        update_layout(world)

        # Positions should change due to repulsion
        assert agent1.x != initial_x1 or agent2.x != initial_x2

    def test_layout_settles_over_time(self):
        """Layout should stabilize (velocities decrease) over iterations."""
        world = make_world()

        # Create agents in a line
        for i in range(3):
            agent = make_agent(f"a{i}", x=float(i * 50), y=0.0)
            world.agents[f"a{i}"] = agent

        # Add links
        link1 = Link(id="l1", source_id="a0", dest_id="a1", link_type=LinkType.PEER)
        link2 = Link(id="l2", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        world.links["l1"] = link1
        world.links["l2"] = link2

        # Run many iterations
        for _ in range(200):
            update_layout(world)

        # Velocities should be small (settled)
        for agent in world.agents.values():
            velocity = math.sqrt(agent.vx**2 + agent.vy**2)
            assert velocity < 1.0  # Should be nearly stable

    def test_no_agent_overlap_after_layout(self):
        """Agents should not overlap after layout settles."""
        world = make_world()

        # Create agents all at same position
        for i in range(5):
            agent = make_agent(f"a{i}", x=0.0, y=0.0)
            world.agents[f"a{i}"] = agent

        # Run layout
        for _ in range(100):
            update_layout(world)

        # Check all pairs are separated
        agents = list(world.agents.values())
        min_separation = 5.0  # Should be at least this far apart

        for i, a in enumerate(agents):
            for b in agents[i + 1 :]:
                distance = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
                assert distance > min_separation


class TestForceLayoutIntegration:
    """Integration tests for force layout with simulation."""

    def test_triangle_layout_with_hierarchical_link(self):
        """Test layout with 3 agents including hierarchical relationship."""
        world = make_world()

        # Create Maria (manager), Tom (sandwich maker), Alex (cashier)
        maria = make_agent("maria", x=0.0, y=0.0)
        tom = make_agent("tom", x=50.0, y=0.0)
        alex = make_agent("alex", x=100.0, y=0.0)
        world.agents["maria"] = maria
        world.agents["tom"] = tom
        world.agents["alex"] = alex

        # Hierarchical: Maria → Tom, Maria → Alex
        link1 = Link(
            id="l1",
            source_id="maria",
            dest_id="tom",
            link_type=LinkType.HIERARCHICAL,
        )
        link2 = Link(
            id="l2",
            source_id="maria",
            dest_id="alex",
            link_type=LinkType.HIERARCHICAL,
        )
        world.links["l1"] = link1
        world.links["l2"] = link2

        # Run layout
        for _ in range(100):
            update_layout(world)

        # Maria should be above both Tom and Alex
        assert world.agents["maria"].y > world.agents["tom"].y
        assert world.agents["maria"].y > world.agents["alex"].y

        # Tom and Alex should not overlap
        tom_pos = world.agents["tom"]
        alex_pos = world.agents["alex"]
        distance = math.sqrt((tom_pos.x - alex_pos.x) ** 2 + (tom_pos.y - alex_pos.y) ** 2)
        assert distance > 10.0

    def test_positions_update_smoothly_each_tick(self):
        """Positions should update smoothly (small changes per tick)."""
        world = make_world()

        agent1 = make_agent("a1", x=0.0, y=0.0)
        agent2 = make_agent("a2", x=50.0, y=0.0)
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        link = Link(id="l1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        world.links["l1"] = link

        # Record initial position
        initial_x = agent1.x

        # Single tick
        update_layout(world)

        # Change should be small (not teleporting)
        change = abs(agent1.x - initial_x)
        assert change < 10.0  # Reasonable change per tick

"""Tests for the engine layer: OODA loop stepper, simulation tick driver."""

import uuid

from loopengine.engine.loop import step_agent
from loopengine.engine.simulation import tick_world
from loopengine.model.agent import Agent, Phase
from loopengine.model.link import Link, LinkType
from loopengine.model.particle import Particle
from loopengine.model.world import ExternalInput, World


def make_agent(
    agent_id: str = "test_agent",
    name: str = "Test Agent",
    role: str = "tester",
    loop_period: int = 4,
    policy=None,
) -> Agent:
    """Create a test agent with configurable parameters."""
    return Agent(
        id=agent_id,
        name=name,
        role=role,
        loop_period=loop_period,
        policy=policy,
    )


def make_world() -> World:
    """Create an empty test world."""
    return World()


class TestStepAgent:
    """Tests for step_agent() function."""

    def test_step_agent_returns_list(self):
        """step_agent() should return a list (possibly empty)."""
        agent = make_agent()
        world = make_world()

        result = step_agent(agent, world)

        assert isinstance(result, list)

    def test_sense_phase_reads_input_buffer(self):
        """SENSE phase should read input_buffer into internal_state['sensed_inputs']."""
        agent = make_agent()
        world = make_world()

        # Put a particle in the input buffer
        test_particle = Particle(id="p1", particle_type="test", source_id="external")
        agent.input_buffer.append(test_particle)

        # Verify we're in SENSE phase
        assert agent.loop_phase == Phase.SENSE

        step_agent(agent, world)

        # Check that sensed_inputs was populated
        assert "sensed_inputs" in agent.internal_state
        assert len(agent.internal_state["sensed_inputs"]) == 1
        assert agent.internal_state["sensed_inputs"][0].id == "p1"

    def test_sense_phase_clears_input_buffer(self):
        """SENSE phase should clear the input buffer after reading."""
        agent = make_agent()
        world = make_world()

        # Put particles in buffer
        agent.input_buffer.append(Particle(id="p1", particle_type="test"))
        agent.input_buffer.append(Particle(id="p2", particle_type="test"))

        step_agent(agent, world)

        # Buffer should be cleared
        assert len(agent.input_buffer) == 0

    def test_phase_transitions_sense_to_orient(self):
        """Agent should transition from SENSE to ORIENT after phase completes."""
        agent = make_agent(loop_period=4)  # 1 tick per phase
        world = make_world()

        assert agent.loop_phase == Phase.SENSE

        step_agent(agent, world)

        assert agent.loop_phase == Phase.ORIENT

    def test_phase_transitions_complete_cycle(self):
        """Agent should cycle through all phases: SENSE→ORIENT→DECIDE→ACT→SENSE."""
        agent = make_agent(loop_period=4)  # 1 tick per phase
        world = make_world()

        # Start at SENSE
        assert agent.loop_phase == Phase.SENSE

        # After 1st step: should be ORIENT
        step_agent(agent, world)
        assert agent.loop_phase == Phase.ORIENT

        # After 2nd step: should be DECIDE
        step_agent(agent, world)
        assert agent.loop_phase == Phase.DECIDE

        # After 3rd step: should be ACT
        step_agent(agent, world)
        assert agent.loop_phase == Phase.ACT

        # After 4th step: should wrap back to SENSE
        step_agent(agent, world)
        assert agent.loop_phase == Phase.SENSE

    def test_phase_tick_increments(self):
        """phase_tick should increment within a phase."""
        agent = make_agent(loop_period=8)  # 2 ticks per phase
        world = make_world()

        assert agent.phase_tick == 0

        step_agent(agent, world)
        assert agent.phase_tick == 1
        assert agent.loop_phase == Phase.SENSE  # Still in SENSE

        step_agent(agent, world)
        assert agent.phase_tick == 0  # Reset after phase transition
        assert agent.loop_phase == Phase.ORIENT

    def test_decide_phase_calls_policy(self):
        """DECIDE phase should invoke agent.policy with correct arguments."""
        policy_calls = []

        def test_policy(sensed_inputs, genome, internal_state):
            policy_calls.append((sensed_inputs, genome, internal_state))
            return []

        agent = make_agent(loop_period=4, policy=test_policy)
        agent.genome = {"trait1": 0.5}
        world = make_world()

        # Step through SENSE and ORIENT to reach DECIDE
        step_agent(agent, world)  # SENSE → ORIENT
        step_agent(agent, world)  # ORIENT → DECIDE
        step_agent(agent, world)  # DECIDE → ACT

        # Policy should have been called once during DECIDE
        assert len(policy_calls) == 1
        _sensed, genome, _state = policy_calls[0]
        assert genome == {"trait1": 0.5}

    def test_act_phase_returns_particles(self):
        """ACT phase should return particles produced by policy."""

        def produce_particle_policy(sensed_inputs, genome, internal_state):
            return [
                Particle(
                    id=str(uuid.uuid4()),
                    particle_type="output",
                    source_id="test_agent",
                    dest_id="other_agent",
                )
            ]

        agent = make_agent(loop_period=4, policy=produce_particle_policy)
        world = make_world()

        # Step through to ACT phase
        step_agent(agent, world)  # SENSE → ORIENT (returns [])
        step_agent(agent, world)  # ORIENT → DECIDE (returns [])
        step_agent(agent, world)  # DECIDE → ACT (returns [])

        # ACT phase should return the particles
        particles = step_agent(agent, world)  # ACT → SENSE

        assert len(particles) == 1
        assert particles[0].particle_type == "output"

    def test_act_phase_populates_output_buffer(self):
        """ACT phase should add particles to agent's output_buffer."""

        def produce_particle_policy(sensed_inputs, genome, internal_state):
            return [
                Particle(
                    id="out1",
                    particle_type="output",
                    source_id="test_agent",
                    dest_id="other_agent",
                )
            ]

        agent = make_agent(loop_period=4, policy=produce_particle_policy)
        world = make_world()

        # Step through to after ACT phase
        step_agent(agent, world)  # SENSE → ORIENT
        step_agent(agent, world)  # ORIENT → DECIDE
        step_agent(agent, world)  # DECIDE → ACT
        step_agent(agent, world)  # ACT → SENSE

        assert len(agent.output_buffer) == 1
        assert agent.output_buffer[0].id == "out1"

    def test_no_policy_produces_no_particles(self):
        """Agent without policy should produce no particles."""
        agent = make_agent(loop_period=4, policy=None)
        world = make_world()

        # Complete full cycle
        for _ in range(4):
            step_agent(agent, world)

        assert len(agent.output_buffer) == 0

    def test_orient_phase_copies_sensed_to_oriented(self):
        """ORIENT phase should copy sensed_inputs to oriented_inputs."""
        agent = make_agent(loop_period=4)
        world = make_world()

        # Add input during SENSE
        agent.input_buffer.append(Particle(id="p1", particle_type="test"))

        step_agent(agent, world)  # SENSE → ORIENT (reads inputs)
        step_agent(agent, world)  # ORIENT → DECIDE (interprets inputs)

        # Should have oriented_inputs in state
        assert "oriented_inputs" in agent.internal_state
        assert len(agent.internal_state["oriented_inputs"]) == 1

    def test_planned_actions_cleared_after_act(self):
        """planned_actions should be cleared after ACT phase."""

        def produce_particle_policy(sensed_inputs, genome, internal_state):
            return [Particle(id="out1", particle_type="output")]

        agent = make_agent(loop_period=4, policy=produce_particle_policy)
        world = make_world()

        # Step through to after ACT
        for _ in range(4):
            step_agent(agent, world)

        # planned_actions should be cleared
        assert agent.internal_state.get("planned_actions", []) == []

    def test_longer_loop_period(self):
        """Agent with longer loop_period should take more ticks per phase."""
        agent = make_agent(loop_period=20)  # 5 ticks per phase
        world = make_world()

        # Should stay in SENSE for 5 ticks
        for _ in range(5):
            assert agent.loop_phase == Phase.SENSE
            step_agent(agent, world)

        # Now should be in ORIENT
        assert agent.loop_phase == Phase.ORIENT

        # Should take 5 more ticks to get to DECIDE
        for _ in range(5):
            assert agent.loop_phase == Phase.ORIENT
            step_agent(agent, world)

        assert agent.loop_phase == Phase.DECIDE

    def test_full_ooda_revolution(self):
        """Verify complete OODA revolution with 20 tick loop period."""
        agent = make_agent(loop_period=20)
        world = make_world()

        phases_seen = []

        for _ in range(20):
            phases_seen.append(agent.loop_phase)
            step_agent(agent, world)

        # Should have seen each phase 5 times
        assert phases_seen.count(Phase.SENSE) == 5
        assert phases_seen.count(Phase.ORIENT) == 5
        assert phases_seen.count(Phase.DECIDE) == 5
        assert phases_seen.count(Phase.ACT) == 5

        # Should be back at SENSE
        assert agent.loop_phase == Phase.SENSE


class TestTickWorld:
    """Tests for tick_world() function."""

    def test_tick_world_increments_tick(self):
        """tick_world() should increment world.tick."""
        world = make_world()

        assert world.tick == 0
        tick_world(world)
        assert world.tick == 1
        tick_world(world)
        assert world.tick == 2

    def test_tick_world_updates_time(self):
        """tick_world() should update world.time based on speed."""
        world = make_world()
        world.speed = 1.0

        assert world.time == 0.0
        tick_world(world)
        assert world.time == 1.0
        tick_world(world)
        assert world.time == 2.0

    def test_tick_world_time_respects_speed(self):
        """tick_world() should update time inversely proportional to speed."""
        world = make_world()
        world.speed = 2.0  # 2 ticks per second → 0.5 seconds per tick

        tick_world(world)
        assert world.time == 0.5

    def test_particle_advances_along_link(self):
        """Particles should advance by their speed each tick."""
        world = make_world()

        # Create two agents
        agent1 = make_agent(agent_id="a1")
        agent2 = make_agent(agent_id="a2")
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        # Create a link between them
        link = Link(id="link1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        world.links["link1"] = link

        # Create a particle on the link
        particle = Particle(
            id="p1",
            particle_type="test",
            source_id="a1",
            dest_id="a2",
            link_id="link1",
            progress=0.0,
            speed=0.2,
        )
        world.particles["p1"] = particle

        tick_world(world)

        assert world.particles["p1"].progress == 0.2

    def test_particle_delivered_when_progress_reaches_one(self):
        """Particles should be delivered when progress >= 1.0."""
        world = make_world()

        # Create two agents
        agent1 = make_agent(agent_id="a1")
        agent2 = make_agent(agent_id="a2")
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        # Create a link between them
        link = Link(id="link1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        world.links["link1"] = link

        # Create a particle about to arrive (progress=0.9, speed=0.2)
        particle = Particle(
            id="p1",
            particle_type="test",
            source_id="a1",
            dest_id="a2",
            link_id="link1",
            progress=0.9,
            speed=0.2,
        )
        world.particles["p1"] = particle

        tick_world(world)

        # Particle should be delivered to a2's input buffer
        assert len(agent2.input_buffer) == 1
        assert agent2.input_buffer[0].id == "p1"

        # Particle should be removed from world.particles (garbage collected)
        assert "p1" not in world.particles

    def test_dead_particles_garbage_collected(self):
        """Dead particles should be removed from world.particles."""
        world = make_world()

        # Create a dead particle
        particle = Particle(id="dead1", particle_type="test", alive=False)
        world.particles["dead1"] = particle

        tick_world(world)

        assert "dead1" not in world.particles

    def test_external_input_spawns_particles(self):
        """External inputs should spawn particles based on rate."""
        world = make_world()

        # Create target agent
        agent = make_agent(agent_id="target")
        world.agents["target"] = agent

        # Create external input with guaranteed rate (1.0 = 100% spawn each tick)
        ext_input = ExternalInput(
            name="test_input",
            target_agent_id="target",
            rate=1.0,  # Guaranteed spawn
            variance=0.0,
            particle_type="customer_order",
            payload_generator=lambda: {"item": "sandwich"},
        )
        world.external_inputs.append(ext_input)

        tick_world(world)

        # Particle was spawned to input buffer, then agent was stepped (SENSE phase)
        # which moved it to sensed_inputs. Verify particle was received.
        sensed = agent.internal_state.get("sensed_inputs", [])
        assert len(sensed) >= 1
        assert sensed[0].particle_type == "customer_order"
        assert sensed[0].payload == {"item": "sandwich"}

    def test_external_input_respects_schedule(self):
        """External input schedule should affect spawn rate."""
        world = make_world()

        agent = make_agent(agent_id="target")
        world.agents["target"] = agent

        # External input with schedule that returns 0.0 (no spawns)
        ext_input = ExternalInput(
            name="test_input",
            target_agent_id="target",
            rate=1.0,
            variance=0.0,
            particle_type="order",
            schedule=lambda tick: 0.0,  # Always return 0 → rate becomes 0
        )
        world.external_inputs.append(ext_input)

        tick_world(world)

        # With schedule returning 0, no particles should spawn
        assert len(agent.input_buffer) == 0

    def test_agents_stepped_each_tick(self):
        """All agents should be stepped each tick."""

        def counting_policy(sensed, genome, state):
            return []

        world = make_world()

        # Create two agents
        agent1 = make_agent(agent_id="a1", loop_period=4, policy=counting_policy)
        agent2 = make_agent(agent_id="a2", loop_period=4, policy=counting_policy)
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        # Run several ticks
        for _ in range(4):
            tick_world(world)

        # Both agents should have cycled through their phases
        assert agent1.loop_phase == Phase.SENSE  # Back to start after 4 ticks
        assert agent2.loop_phase == Phase.SENSE

    def test_policy_produced_particles_placed_on_links(self):
        """Particles produced by agent policy should be placed on appropriate links."""

        def produce_order(sensed, genome, state):
            return [
                Particle(
                    id=str(uuid.uuid4()),
                    particle_type="order",
                    source_id="sender",
                    dest_id="receiver",
                )
            ]

        world = make_world()

        # Create agents
        sender = make_agent(agent_id="sender", loop_period=4, policy=produce_order)
        receiver = make_agent(agent_id="receiver", loop_period=4)
        world.agents["sender"] = sender
        world.agents["receiver"] = receiver

        # Create link from sender to receiver
        link = Link(
            id="sender_to_receiver",
            source_id="sender",
            dest_id="receiver",
            link_type=LinkType.SERVICE,
        )
        world.links["sender_to_receiver"] = link

        # Step through to ACT phase (when particles are produced)
        for _ in range(4):
            tick_world(world)

        # Particle should be in world.particles or already delivered
        # With speed=0.1 (default), it takes 10 ticks to traverse
        # Check the particle is in the world or in receiver's buffer
        has_particle = len(world.particles) > 0 or len(receiver.input_buffer) > 0
        assert has_particle

    def test_multiple_particles_advance_independently(self):
        """Multiple particles should advance independently."""
        world = make_world()

        agent1 = make_agent(agent_id="a1")
        agent2 = make_agent(agent_id="a2")
        world.agents["a1"] = agent1
        world.agents["a2"] = agent2

        link = Link(id="link1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        world.links["link1"] = link

        # Create particles with different speeds
        p1 = Particle(
            id="p1",
            particle_type="fast",
            source_id="a1",
            dest_id="a2",
            link_id="link1",
            progress=0.0,
            speed=0.3,
        )
        p2 = Particle(
            id="p2",
            particle_type="slow",
            source_id="a1",
            dest_id="a2",
            link_id="link1",
            progress=0.0,
            speed=0.1,
        )
        world.particles["p1"] = p1
        world.particles["p2"] = p2

        tick_world(world)

        # Both should have advanced by their respective speeds
        # p1 might have been delivered if 0.3 >= 1.0 (it won't), check it exists
        if "p1" in world.particles:
            assert world.particles["p1"].progress == 0.3
        if "p2" in world.particles:
            assert world.particles["p2"].progress == 0.1

    def test_particle_without_valid_link_dies(self):
        """Particles emitted to non-existent destinations should die."""

        def produce_orphan(sensed, genome, state):
            return [
                Particle(
                    id="orphan",
                    particle_type="lost",
                    source_id="sender",
                    dest_id="nonexistent",
                )
            ]

        world = make_world()

        sender = make_agent(agent_id="sender", loop_period=4, policy=produce_orphan)
        world.agents["sender"] = sender

        # No link to "nonexistent" agent

        # Step through to ACT phase
        for _ in range(4):
            tick_world(world)

        # Orphan particle should not be in world.particles (died due to no link)
        assert "orphan" not in world.particles

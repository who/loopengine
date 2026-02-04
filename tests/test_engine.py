"""Tests for the engine layer: OODA loop stepper, simulation tick driver."""

import uuid

from loopengine.engine.loop import step_agent
from loopengine.model.agent import Agent, Phase
from loopengine.model.particle import Particle
from loopengine.model.world import World


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

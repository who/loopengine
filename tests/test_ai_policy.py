"""Tests for AI policy integration with simulation engine."""

from unittest.mock import MagicMock

from loopengine.behaviors import (
    AIBehaviorEngine,
    BehaviorResponse,
    DomainContext,
)
from loopengine.engine.ai_policy import (
    ActionConverter,
    AIPolicy,
    create_ai_policy_for_agent,
    enable_ai_behaviors,
)
from loopengine.engine.loop import step_agent
from loopengine.engine.simulation import tick_world
from loopengine.model.agent import Agent
from loopengine.model.link import Link, LinkType
from loopengine.model.particle import Particle
from loopengine.model.world import World


def make_agent(
    agent_id: str = "test_agent",
    name: str = "Test Agent",
    role: str = "tester",
    loop_period: int = 4,
    policy=None,
    labels: set[str] | None = None,
) -> Agent:
    """Create a test agent with configurable parameters."""
    return Agent(
        id=agent_id,
        name=name,
        role=role,
        loop_period=loop_period,
        policy=policy,
        labels=labels or set(),
    )


def make_world() -> World:
    """Create an empty test world."""
    return World()


def make_test_world_with_links() -> tuple[World, Agent, Agent]:
    """Create a test world with two agents and a link between them."""
    world = make_world()

    sender = make_agent(agent_id="sender", name="Sender", role="producer")
    receiver = make_agent(agent_id="receiver", name="Receiver", role="consumer")
    world.agents["sender"] = sender
    world.agents["receiver"] = receiver

    link = Link(
        id="sender_to_receiver",
        source_id="sender",
        dest_id="receiver",
        link_type=LinkType.SERVICE,
    )
    world.links["sender_to_receiver"] = link

    return world, sender, receiver


class TestActionConverter:
    """Tests for ActionConverter class."""

    def test_idle_action_returns_empty_list(self):
        """Idle/wait actions should produce no particles."""
        converter = ActionConverter()
        world, sender, _ = make_test_world_with_links()

        for action in ("idle", "wait", "observe", "do_nothing"):
            response = BehaviorResponse(
                action=action,
                parameters={},
                reasoning="Nothing to do",
                metadata={},
            )
            result = converter.convert(response, sender, world)
            assert result == []

    def test_explicit_particle_params(self):
        """Actions with explicit particle_type and dest_id should create particles."""
        converter = ActionConverter()
        world, sender, _ = make_test_world_with_links()

        response = BehaviorResponse(
            action="custom_action",
            parameters={
                "particle_type": "order",
                "dest_id": "receiver",
                "payload": {"item": "test"},
            },
            reasoning="Creating custom particle",
            metadata={},
        )

        result = converter.convert(response, sender, world)

        assert len(result) == 1
        assert result[0].particle_type == "order"
        assert result[0].dest_id == "receiver"
        assert result[0].payload == {"item": "test"}
        assert result[0].source_id == "sender"
        assert result[0].link_id == "sender_to_receiver"

    def test_send_action_pattern(self):
        """send_* actions should create appropriately typed particles."""
        converter = ActionConverter()
        world, sender, _ = make_test_world_with_links()

        response = BehaviorResponse(
            action="send_sandwich",
            parameters={"dest_id": "receiver", "quality": 0.9},
            reasoning="Sending sandwich",
            metadata={},
        )

        result = converter.convert(response, sender, world)

        assert len(result) == 1
        assert result[0].particle_type == "sandwich"
        assert result[0].dest_id == "receiver"

    def test_production_action(self):
        """Production actions should create finished product particles."""
        converter = ActionConverter()
        world, sender, _ = make_test_world_with_links()

        response = BehaviorResponse(
            action="make_sandwich",
            parameters={"order": {"type": "BLT"}, "quality": 0.85},
            reasoning="Making sandwich",
            metadata={},
        )

        result = converter.convert(response, sender, world)

        assert len(result) == 1
        assert result[0].particle_type == "finished_sandwich"
        assert result[0].payload["order"] == {"type": "BLT"}
        assert result[0].payload["quality"] == 0.85
        assert result[0].payload["maker"] == "sender"

    def test_ticket_action(self):
        """Ticket creation actions should create order_ticket particles."""
        converter = ActionConverter()
        world, sender, _ = make_test_world_with_links()

        response = BehaviorResponse(
            action="create_ticket",
            parameters={"order": {"sandwich": "Club"}},
            reasoning="Creating order ticket",
            metadata={},
        )

        result = converter.convert(response, sender, world)

        assert len(result) == 1
        assert result[0].particle_type == "order_ticket"

    def test_serve_action(self):
        """Serve actions should create served_customer particles to external."""
        converter = ActionConverter()
        world, sender, _ = make_test_world_with_links()

        response = BehaviorResponse(
            action="serve_customer",
            parameters={"customer_id": "cust123"},
            reasoning="Serving customer",
            metadata={},
        )

        result = converter.convert(response, sender, world)

        assert len(result) == 1
        assert result[0].particle_type == "served_customer"
        assert result[0].dest_id == "external"
        assert result[0].link_id == ""

    def test_generic_action(self):
        """Unknown actions should create particles with action as type."""
        converter = ActionConverter()
        world, sender, _ = make_test_world_with_links()

        response = BehaviorResponse(
            action="unknown_action",
            parameters={"data": "value"},
            reasoning="Doing something",
            metadata={},
        )

        result = converter.convert(response, sender, world)

        assert len(result) == 1
        assert result[0].particle_type == "unknown_action"

    def test_link_lookup(self):
        """Converter should find appropriate links for routing."""
        converter = ActionConverter()
        world, sender, _ = make_test_world_with_links()

        # Add reverse link
        world.links["receiver_to_sender"] = Link(
            id="receiver_to_sender",
            source_id="receiver",
            dest_id="sender",
            link_type=LinkType.SERVICE,
        )

        # Action from sender to receiver should use sender_to_receiver link
        response = BehaviorResponse(
            action="send_data",
            parameters={"dest_id": "receiver"},
            reasoning="Sending data",
            metadata={},
        )

        result = converter.convert(response, sender, world)
        assert result[0].link_id == "sender_to_receiver"


class TestAIPolicy:
    """Tests for AIPolicy class."""

    def test_create_callable_returns_callable(self):
        """create_callable should return a callable function."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="idle",
            parameters={},
            reasoning="Nothing to do",
            metadata={},
        )

        domain = DomainContext(domain_type="test", domain_description="Test domain")
        policy = AIPolicy(engine=mock_engine, domain=domain)

        world, sender, _ = make_test_world_with_links()
        callable_policy = policy.create_callable(sender, world)

        assert callable(callable_policy)

    def test_policy_calls_engine_generate_behavior(self):
        """Policy should call engine.generate_behavior with correct context."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="idle",
            parameters={},
            reasoning="Waiting",
            metadata={"latency_ms": 100},
        )

        domain = DomainContext(domain_type="sandwich_shop", domain_description="A shop")
        policy = AIPolicy(engine=mock_engine, domain=domain)

        world, sender, _ = make_test_world_with_links()
        sender.genome = {"speed": 0.8}
        sender.labels = {"Kitchen", "FrontLine"}

        callable_policy = policy.create_callable(sender, world)

        # Call the policy
        sensed_inputs = [Particle(id="p1", particle_type="order", source_id="external")]
        callable_policy(sensed_inputs, sender.genome, sender.internal_state)

        # Verify engine was called
        mock_engine.generate_behavior.assert_called_once()
        call_args = mock_engine.generate_behavior.call_args

        # Check domain was passed
        assert call_args.kwargs["domain"] == domain

        # Check agent context
        agent_ctx = call_args.kwargs["agent"]
        assert agent_ctx.agent_type == "producer"

        # Check context contains expected data
        context = call_args.kwargs["context"]
        assert context["agent_id"] == "sender"
        assert len(context["sensed_inputs"]) == 1
        assert context["genome_traits"] == {"speed": 0.8}

    def test_policy_converts_response_to_particles(self):
        """Policy should convert LLM response to Particle outputs."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="send_product",
            parameters={"dest_id": "receiver"},
            reasoning="Sending product",
            metadata={},
        )

        domain = DomainContext(domain_type="test", domain_description="Test")
        policy = AIPolicy(engine=mock_engine, domain=domain)

        world, sender, _ = make_test_world_with_links()
        callable_policy = policy.create_callable(sender, world)

        result = callable_policy([], sender.genome, sender.internal_state)

        assert len(result) == 1
        assert isinstance(result[0], Particle)
        assert result[0].particle_type == "product"

    def test_policy_stores_response_in_internal_state(self):
        """Policy should store last AI response in internal state."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="test_action",
            parameters={"key": "value"},
            reasoning="Test reasoning",
            metadata={"latency_ms": 50},
        )

        domain = DomainContext(domain_type="test", domain_description="Test")
        policy = AIPolicy(engine=mock_engine, domain=domain)

        world, sender, _ = make_test_world_with_links()
        callable_policy = policy.create_callable(sender, world)

        internal_state = {}
        callable_policy([], {}, internal_state)

        assert "last_ai_response" in internal_state
        assert internal_state["last_ai_response"]["action"] == "test_action"
        assert internal_state["last_ai_response"]["reasoning"] == "Test reasoning"

    def test_policy_uses_fallback_on_error(self):
        """Policy should use fallback policy when engine fails."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.side_effect = Exception("LLM error")

        fallback_outputs = [Particle(id="fallback", particle_type="fallback_action")]
        fallback_policy = MagicMock(return_value=fallback_outputs)

        domain = DomainContext(domain_type="test", domain_description="Test")
        policy = AIPolicy(engine=mock_engine, domain=domain, fallback_policy=fallback_policy)

        world, sender, _ = make_test_world_with_links()
        callable_policy = policy.create_callable(sender, world)

        result = callable_policy([], {}, {})

        assert len(result) == 1
        assert result[0].id == "fallback"
        fallback_policy.assert_called_once()

    def test_policy_returns_empty_on_error_without_fallback(self):
        """Policy should return empty list on error without fallback."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.side_effect = Exception("LLM error")

        domain = DomainContext(domain_type="test", domain_description="Test")
        policy = AIPolicy(engine=mock_engine, domain=domain)

        world, sender, _ = make_test_world_with_links()
        callable_policy = policy.create_callable(sender, world)

        result = callable_policy([], {}, {})

        assert result == []


class TestCreateAIPolicyForAgent:
    """Tests for create_ai_policy_for_agent helper function."""

    def test_creates_callable_policy(self):
        """create_ai_policy_for_agent should return a callable."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="idle",
            parameters={},
            reasoning="Idle",
            metadata={},
        )

        domain = DomainContext(domain_type="test", domain_description="Test")
        world, sender, _ = make_test_world_with_links()

        policy = create_ai_policy_for_agent(sender, world, mock_engine, domain)

        assert callable(policy)

        # Verify it works
        result = policy([], {}, {})
        assert isinstance(result, list)


class TestEnableAIBehaviors:
    """Tests for enable_ai_behaviors function."""

    def test_enables_ai_for_all_agents(self):
        """enable_ai_behaviors should replace policies for all agents."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="idle",
            parameters={},
            reasoning="Idle",
            metadata={},
        )

        domain = DomainContext(domain_type="test", domain_description="Test")
        world, sender, receiver = make_test_world_with_links()

        original_sender_policy = sender.policy
        original_receiver_policy = receiver.policy

        enable_ai_behaviors(world, mock_engine, domain)

        # Both agents should have new policies
        assert sender.policy is not original_sender_policy
        assert receiver.policy is not original_receiver_policy
        assert callable(sender.policy)
        assert callable(receiver.policy)

        # Both should be marked as AI-enabled
        assert sender.internal_state.get("ai_enabled") is True
        assert receiver.internal_state.get("ai_enabled") is True

    def test_enables_ai_for_specific_agent_types(self):
        """enable_ai_behaviors should filter by agent_types."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="idle",
            parameters={},
            reasoning="Idle",
            metadata={},
        )

        domain = DomainContext(domain_type="test", domain_description="Test")
        world, sender, receiver = make_test_world_with_links()

        # Only enable for "producer" role
        enable_ai_behaviors(world, mock_engine, domain, agent_types={"producer"})

        # Only sender (producer) should be AI-enabled
        assert sender.internal_state.get("ai_enabled") is True
        assert receiver.internal_state.get("ai_enabled") is None

    def test_preserves_fallback_policy(self):
        """enable_ai_behaviors should preserve existing policy as fallback."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        # Engine fails on first call, should trigger fallback
        mock_engine.generate_behavior.side_effect = Exception("LLM error")

        original_output = [Particle(id="original", particle_type="original")]

        def original_policy(sensed, genome, state):
            return original_output

        domain = DomainContext(domain_type="test", domain_description="Test")
        world, sender, _ = make_test_world_with_links()
        sender.policy = original_policy

        enable_ai_behaviors(world, mock_engine, domain, preserve_fallback=True)

        # Call the new policy - should fall back to original
        result = sender.policy([], {}, {})

        assert len(result) == 1
        assert result[0].id == "original"

    def test_no_fallback_when_preserve_false(self):
        """enable_ai_behaviors with preserve_fallback=False should not keep fallback."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.side_effect = Exception("LLM error")

        def original_policy(sensed, genome, state):
            return [Particle(id="original", particle_type="original")]

        domain = DomainContext(domain_type="test", domain_description="Test")
        world, sender, _ = make_test_world_with_links()
        sender.policy = original_policy

        enable_ai_behaviors(world, mock_engine, domain, preserve_fallback=False)

        # Call the new policy - should return empty (no fallback)
        result = sender.policy([], {}, {})

        assert result == []


class TestAIPolicyIntegrationWithSimulation:
    """Integration tests for AI policy with simulation loop."""

    def test_ai_policy_works_in_step_agent(self):
        """AI policy should work correctly within step_agent loop."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="send_result",
            parameters={"dest_id": "receiver", "value": 42},
            reasoning="Sending result",
            metadata={"latency_ms": 100},
        )

        domain = DomainContext(domain_type="test", domain_description="Test")
        world, sender, _receiver = make_test_world_with_links()

        # Set up AI policy
        enable_ai_behaviors(world, mock_engine, domain, agent_types={"producer"})

        # Add input particle
        sender.input_buffer.append(Particle(id="input1", particle_type="request"))

        # Step through OODA cycle
        step_agent(sender, world)  # SENSE
        step_agent(sender, world)  # ORIENT
        step_agent(sender, world)  # DECIDE - calls AI policy
        particles = step_agent(sender, world)  # ACT - returns particles

        # Should have produced output
        assert len(particles) == 1
        assert particles[0].particle_type == "result"

    def test_ai_policy_works_in_tick_world(self):
        """AI policy should work within full tick_world simulation."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="idle",
            parameters={},
            reasoning="Waiting",
            metadata={},
        )

        domain = DomainContext(domain_type="test", domain_description="Test")
        world, _sender, _receiver = make_test_world_with_links()

        enable_ai_behaviors(world, mock_engine, domain)

        # Run several ticks
        for _ in range(8):
            tick_world(world)

        # Verify world ticks advanced
        assert world.tick == 8

        # Verify AI was called (once per agent per DECIDE phase)
        # With loop_period=4, each agent does 2 full cycles in 8 ticks
        # That's 2 DECIDE phases per agent = 4 calls total
        assert mock_engine.generate_behavior.call_count == 4

    def test_context_includes_sensed_inputs(self):
        """AI policy context should include particles sensed by agent."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="idle",
            parameters={},
            reasoning="Processed",
            metadata={},
        )

        domain = DomainContext(domain_type="test", domain_description="Test")
        world, sender, _ = make_test_world_with_links()

        enable_ai_behaviors(world, mock_engine, domain, agent_types={"producer"})

        # Add input to sense
        sender.input_buffer.append(
            Particle(id="order1", particle_type="order", payload={"item": "sandwich"})
        )

        # Step through to DECIDE phase
        step_agent(sender, world)  # SENSE
        step_agent(sender, world)  # ORIENT
        step_agent(sender, world)  # DECIDE - calls AI

        # Check that context included the sensed input
        call_args = mock_engine.generate_behavior.call_args
        context = call_args.kwargs["context"]

        assert len(context["sensed_inputs"]) == 1
        assert context["sensed_inputs"][0]["type"] == "order"
        assert context["sensed_inputs"][0]["payload"] == {"item": "sandwich"}

    def test_particles_routed_correctly(self):
        """Particles from AI policy should be properly routed to destinations."""
        mock_engine = MagicMock(spec=AIBehaviorEngine)
        mock_engine.generate_behavior.return_value = BehaviorResponse(
            action="send_product",
            parameters={"dest_id": "receiver"},
            reasoning="Sending",
            metadata={},
        )

        domain = DomainContext(domain_type="test", domain_description="Test")
        world, _sender, receiver = make_test_world_with_links()

        enable_ai_behaviors(world, mock_engine, domain, agent_types={"producer"})

        # Track all sensed particles across ticks
        all_sensed: list[Particle] = []

        # Run until particle is produced, delivered, and sensed
        # 4 ticks to produce + 10 ticks to traverse link + 4 ticks to sense = ~18 ticks
        for _ in range(20):
            tick_world(world)
            # Collect sensed inputs after each tick
            sensed = receiver.internal_state.get("sensed_inputs", [])
            all_sensed.extend(sensed)

        # Receiver should have received and sensed the particle at some point
        assert len(all_sensed) > 0
        assert any(p.particle_type == "product" for p in all_sensed)

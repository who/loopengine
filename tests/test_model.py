"""Tests for the model layer: Agent, Link, Particle, Label, GenomeSchema, World."""

from datetime import datetime

from loopengine.model import (
    Agent,
    ExternalInput,
    GenomeSchema,
    GenomeTrait,
    Label,
    LabelContext,
    Link,
    LinkType,
    Particle,
    Phase,
    World,
)


class TestPhase:
    """Tests for Phase enum."""

    def test_phase_values(self):
        """Phase enum should have correct values."""
        assert Phase.SENSE.value == "sense"
        assert Phase.ORIENT.value == "orient"
        assert Phase.DECIDE.value == "decide"
        assert Phase.ACT.value == "act"

    def test_phase_count(self):
        """Phase enum should have exactly 4 phases (OODA)."""
        assert len(Phase) == 4

    def test_phase_iteration(self):
        """Phase enum should be iterable in OODA order."""
        phases = list(Phase)
        assert phases == [Phase.SENSE, Phase.ORIENT, Phase.DECIDE, Phase.ACT]


class TestAgent:
    """Tests for Agent dataclass."""

    def test_agent_creation_with_required_fields(self):
        """Agent should instantiate with required fields."""
        agent = Agent(id="a1", name="Test Agent", role="tester")

        assert agent.id == "a1"
        assert agent.name == "Test Agent"
        assert agent.role == "tester"

    def test_agent_default_genome(self):
        """Agent should have empty genome by default."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.genome == {}

    def test_agent_default_labels(self):
        """Agent should have empty labels set by default."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.labels == set()
        assert isinstance(agent.labels, set)

    def test_agent_default_internal_state(self):
        """Agent should have empty internal_state by default."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.internal_state == {}

    def test_agent_default_input_buffer(self):
        """Agent should have empty input_buffer by default."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.input_buffer == []

    def test_agent_default_output_buffer(self):
        """Agent should have empty output_buffer by default."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.output_buffer == []

    def test_agent_default_loop_period(self):
        """Agent should have default loop_period of 60."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.loop_period == 60

    def test_agent_default_loop_phase(self):
        """Agent should start in SENSE phase."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.loop_phase == Phase.SENSE

    def test_agent_default_phase_tick(self):
        """Agent should have phase_tick of 0 by default."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.phase_tick == 0

    def test_agent_default_policy(self):
        """Agent should have None policy by default."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.policy is None

    def test_agent_default_position(self):
        """Agent should have default position at origin."""
        agent = Agent(id="a1", name="Test", role="test")

        assert agent.x == 0.0
        assert agent.y == 0.0
        assert agent.vx == 0.0
        assert agent.vy == 0.0

    def test_agent_with_genome(self):
        """Agent should accept genome dict."""
        genome = {"speed": 0.8, "accuracy": 0.9}
        agent = Agent(id="a1", name="Test", role="test", genome=genome)

        assert agent.genome == {"speed": 0.8, "accuracy": 0.9}

    def test_agent_with_labels(self):
        """Agent should accept labels set."""
        agent = Agent(id="a1", name="Test", role="test", labels={"kitchen", "staff"})

        assert "kitchen" in agent.labels
        assert "staff" in agent.labels

    def test_agent_with_custom_loop_period(self):
        """Agent should accept custom loop_period."""
        agent = Agent(id="a1", name="Test", role="test", loop_period=30)

        assert agent.loop_period == 30

    def test_agent_with_policy(self):
        """Agent should accept callable policy."""

        def my_policy(sensed, genome, state):
            return []

        agent = Agent(id="a1", name="Test", role="test", policy=my_policy)

        assert agent.policy is my_policy
        assert callable(agent.policy)

    def test_agent_with_position(self):
        """Agent should accept position and velocity."""
        agent = Agent(id="a1", name="Test", role="test", x=10.0, y=20.0, vx=1.0, vy=-1.0)

        assert agent.x == 10.0
        assert agent.y == 20.0
        assert agent.vx == 1.0
        assert agent.vy == -1.0

    def test_agent_input_buffer_is_mutable(self):
        """Agent input_buffer should be mutable list."""
        agent = Agent(id="a1", name="Test", role="test")
        particle = Particle(id="p1", particle_type="test")

        agent.input_buffer.append(particle)

        assert len(agent.input_buffer) == 1
        assert agent.input_buffer[0].id == "p1"

    def test_agent_output_buffer_is_mutable(self):
        """Agent output_buffer should be mutable list."""
        agent = Agent(id="a1", name="Test", role="test")
        particle = Particle(id="p1", particle_type="test")

        agent.output_buffer.append(particle)

        assert len(agent.output_buffer) == 1

    def test_agent_internal_state_is_mutable(self):
        """Agent internal_state should be mutable dict."""
        agent = Agent(id="a1", name="Test", role="test")

        agent.internal_state["key"] = "value"

        assert agent.internal_state["key"] == "value"

    def test_multiple_agents_have_independent_buffers(self):
        """Different agents should not share buffer instances."""
        agent1 = Agent(id="a1", name="Agent 1", role="test")
        agent2 = Agent(id="a2", name="Agent 2", role="test")

        agent1.input_buffer.append(Particle(id="p1", particle_type="test"))

        assert len(agent1.input_buffer) == 1
        assert len(agent2.input_buffer) == 0


class TestLinkType:
    """Tests for LinkType enum."""

    def test_link_type_values(self):
        """LinkType enum should have correct values."""
        assert LinkType.HIERARCHICAL.value == "hierarchical"
        assert LinkType.PEER.value == "peer"
        assert LinkType.SERVICE.value == "service"
        assert LinkType.COMPETITIVE.value == "competitive"

    def test_link_type_count(self):
        """LinkType enum should have 4 types."""
        assert len(LinkType) == 4


class TestLink:
    """Tests for Link dataclass."""

    def test_link_creation_with_required_fields(self):
        """Link should instantiate with required fields."""
        link = Link(id="l1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)

        assert link.id == "l1"
        assert link.source_id == "a1"
        assert link.dest_id == "a2"
        assert link.link_type == LinkType.PEER

    def test_link_default_properties(self):
        """Link should have empty properties dict by default."""
        link = Link(id="l1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)

        assert link.properties == {}

    def test_link_hierarchical_type(self):
        """Link can be hierarchical type."""
        link = Link(id="l1", source_id="boss", dest_id="worker", link_type=LinkType.HIERARCHICAL)

        assert link.link_type == LinkType.HIERARCHICAL

    def test_link_service_type(self):
        """Link can be service type."""
        link = Link(id="l1", source_id="provider", dest_id="consumer", link_type=LinkType.SERVICE)

        assert link.link_type == LinkType.SERVICE

    def test_link_competitive_type(self):
        """Link can be competitive type."""
        link = Link(id="l1", source_id="a1", dest_id="a2", link_type=LinkType.COMPETITIVE)

        assert link.link_type == LinkType.COMPETITIVE

    def test_link_with_properties(self):
        """Link should accept properties dict."""
        props = {
            "authority_scope": ["scheduling", "hiring"],
            "autonomy_granted": 0.7,
            "flow_types": ["directive", "report"],
            "bandwidth": 5.0,
            "latency": 2,
        }
        link = Link(
            id="l1", source_id="a1", dest_id="a2", link_type=LinkType.HIERARCHICAL, properties=props
        )

        assert link.properties["authority_scope"] == ["scheduling", "hiring"]
        assert link.properties["autonomy_granted"] == 0.7
        assert link.properties["flow_types"] == ["directive", "report"]
        assert link.properties["bandwidth"] == 5.0
        assert link.properties["latency"] == 2

    def test_link_properties_is_mutable(self):
        """Link properties should be mutable dict."""
        link = Link(id="l1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)

        link.properties["new_key"] = "new_value"

        assert link.properties["new_key"] == "new_value"

    def test_multiple_links_have_independent_properties(self):
        """Different links should not share properties dict."""
        link1 = Link(id="l1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        link2 = Link(id="l2", source_id="a2", dest_id="a3", link_type=LinkType.PEER)

        link1.properties["key"] = "value1"
        link2.properties["key"] = "value2"

        assert link1.properties["key"] == "value1"
        assert link2.properties["key"] == "value2"


class TestParticle:
    """Tests for Particle dataclass."""

    def test_particle_creation_with_required_fields(self):
        """Particle should instantiate with required fields."""
        particle = Particle(id="p1", particle_type="order")

        assert particle.id == "p1"
        assert particle.particle_type == "order"

    def test_particle_default_payload(self):
        """Particle should have empty payload dict by default."""
        particle = Particle(id="p1", particle_type="test")

        assert particle.payload == {}

    def test_particle_default_source_id(self):
        """Particle should have empty source_id by default."""
        particle = Particle(id="p1", particle_type="test")

        assert particle.source_id == ""

    def test_particle_default_dest_id(self):
        """Particle should have empty dest_id by default."""
        particle = Particle(id="p1", particle_type="test")

        assert particle.dest_id == ""

    def test_particle_default_link_id(self):
        """Particle should have empty link_id by default."""
        particle = Particle(id="p1", particle_type="test")

        assert particle.link_id == ""

    def test_particle_default_progress(self):
        """Particle should have default progress of 0.0."""
        particle = Particle(id="p1", particle_type="test")

        assert particle.progress == 0.0

    def test_particle_default_speed(self):
        """Particle should have default speed of 0.1."""
        particle = Particle(id="p1", particle_type="test")

        assert particle.speed == 0.1

    def test_particle_default_alive(self):
        """Particle should be alive by default."""
        particle = Particle(id="p1", particle_type="test")

        assert particle.alive is True

    def test_particle_with_payload(self):
        """Particle should accept payload dict."""
        payload = {"item": "BLT", "extras": ["bacon", "cheese"]}
        particle = Particle(id="p1", particle_type="order", payload=payload)

        assert particle.payload["item"] == "BLT"
        assert particle.payload["extras"] == ["bacon", "cheese"]

    def test_particle_with_routing(self):
        """Particle should accept routing fields."""
        particle = Particle(
            id="p1",
            particle_type="order",
            source_id="alex",
            dest_id="tom",
            link_id="alex_to_tom",
        )

        assert particle.source_id == "alex"
        assert particle.dest_id == "tom"
        assert particle.link_id == "alex_to_tom"

    def test_particle_with_progress(self):
        """Particle should accept custom progress."""
        particle = Particle(id="p1", particle_type="test", progress=0.5)

        assert particle.progress == 0.5

    def test_particle_progress_zero(self):
        """Particle with zero progress is at source."""
        particle = Particle(id="p1", particle_type="test", progress=0.0)

        assert particle.progress == 0.0

    def test_particle_progress_one(self):
        """Particle with progress 1.0 is at destination."""
        particle = Particle(id="p1", particle_type="test", progress=1.0)

        assert particle.progress == 1.0

    def test_particle_with_speed(self):
        """Particle should accept custom speed."""
        particle = Particle(id="p1", particle_type="test", speed=0.25)

        assert particle.speed == 0.25

    def test_particle_speed_zero(self):
        """Particle can have zero speed (stationary)."""
        particle = Particle(id="p1", particle_type="test", speed=0.0)

        assert particle.speed == 0.0

    def test_particle_dead(self):
        """Particle can be created dead."""
        particle = Particle(id="p1", particle_type="test", alive=False)

        assert particle.alive is False

    def test_particle_payload_is_mutable(self):
        """Particle payload should be mutable."""
        particle = Particle(id="p1", particle_type="test")

        particle.payload["key"] = "value"

        assert particle.payload["key"] == "value"

    def test_multiple_particles_have_independent_payloads(self):
        """Different particles should not share payload dicts."""
        p1 = Particle(id="p1", particle_type="test")
        p2 = Particle(id="p2", particle_type="test")

        p1.payload["key"] = "value1"
        p2.payload["key"] = "value2"

        assert p1.payload["key"] == "value1"
        assert p2.payload["key"] == "value2"


class TestLabelContext:
    """Tests for LabelContext dataclass."""

    def test_label_context_creation_empty(self):
        """LabelContext should instantiate with no arguments."""
        context = LabelContext()

        assert context.constraints == []
        assert context.resources == []
        assert context.norms == []
        assert context.description == ""

    def test_label_context_with_constraints(self):
        """LabelContext should accept constraints list."""
        context = LabelContext(constraints=["health code", "food safety"])

        assert context.constraints == ["health code", "food safety"]

    def test_label_context_with_resources(self):
        """LabelContext should accept resources list."""
        context = LabelContext(resources=["grill", "prep station"])

        assert context.resources == ["grill", "prep station"]

    def test_label_context_with_norms(self):
        """LabelContext should accept norms list."""
        context = LabelContext(norms=["FIFO", "cleanliness"])

        assert context.norms == ["FIFO", "cleanliness"]

    def test_label_context_with_description(self):
        """LabelContext should accept description."""
        context = LabelContext(description="Kitchen workspace context")

        assert context.description == "Kitchen workspace context"

    def test_label_context_full(self):
        """LabelContext should accept all fields."""
        context = LabelContext(
            constraints=["constraint1"],
            resources=["resource1"],
            norms=["norm1"],
            description="Full context",
        )

        assert context.constraints == ["constraint1"]
        assert context.resources == ["resource1"]
        assert context.norms == ["norm1"]
        assert context.description == "Full context"

    def test_label_context_lists_are_mutable(self):
        """LabelContext lists should be mutable."""
        context = LabelContext()

        context.constraints.append("new_constraint")

        assert context.constraints == ["new_constraint"]


class TestLabel:
    """Tests for Label dataclass."""

    def test_label_creation_with_name(self):
        """Label should instantiate with name."""
        label = Label(name="Kitchen")

        assert label.name == "Kitchen"

    def test_label_default_context(self):
        """Label should have empty LabelContext by default."""
        label = Label(name="Kitchen")

        assert isinstance(label.context, LabelContext)
        assert label.context.constraints == []
        assert label.context.resources == []
        assert label.context.norms == []

    def test_label_with_context(self):
        """Label should accept LabelContext."""
        context = LabelContext(
            constraints=["health code"],
            resources=["grill", "fryer"],
            norms=["FIFO"],
        )
        label = Label(name="Kitchen", context=context)

        assert label.context.constraints == ["health code"]
        assert label.context.resources == ["grill", "fryer"]
        assert label.context.norms == ["FIFO"]


class TestGenomeTrait:
    """Tests for GenomeTrait dataclass."""

    def test_genome_trait_creation(self):
        """GenomeTrait should instantiate with name and description."""
        trait = GenomeTrait(name="speed", description="How fast the agent works")

        assert trait.name == "speed"
        assert trait.description == "How fast the agent works"

    def test_genome_trait_default_range(self):
        """GenomeTrait should have default range [0.0, 1.0]."""
        trait = GenomeTrait(name="speed", description="Speed trait")

        assert trait.min_val == 0.0
        assert trait.max_val == 1.0

    def test_genome_trait_default_category(self):
        """GenomeTrait should have empty category by default."""
        trait = GenomeTrait(name="speed", description="Speed trait")

        assert trait.category == ""

    def test_genome_trait_default_discovered_at(self):
        """GenomeTrait should have discovered_at set to current time."""
        before = datetime.now()
        trait = GenomeTrait(name="speed", description="Speed trait")
        after = datetime.now()

        assert before <= trait.discovered_at <= after

    def test_genome_trait_with_custom_range(self):
        """GenomeTrait should accept custom range."""
        trait = GenomeTrait(
            name="temperature", description="Operating temperature", min_val=-10.0, max_val=100.0
        )

        assert trait.min_val == -10.0
        assert trait.max_val == 100.0

    def test_genome_trait_with_category(self):
        """GenomeTrait should accept category."""
        trait = GenomeTrait(name="dexterity", description="Manual skill", category="physical")

        assert trait.category == "physical"

    def test_genome_trait_categories(self):
        """GenomeTrait can have various categories."""
        categories = ["physical", "cognitive", "social", "temperamental", "skill"]

        for cat in categories:
            trait = GenomeTrait(name="test", description="test", category=cat)
            assert trait.category == cat


class TestGenomeSchema:
    """Tests for GenomeSchema dataclass."""

    def test_genome_schema_creation(self):
        """GenomeSchema should instantiate with role."""
        schema = GenomeSchema(role="sandwich_maker")

        assert schema.role == "sandwich_maker"

    def test_genome_schema_default_traits(self):
        """GenomeSchema should have empty traits dict by default."""
        schema = GenomeSchema(role="test")

        assert schema.traits == {}

    def test_genome_schema_default_discovered_at(self):
        """GenomeSchema should have discovered_at set to current time."""
        before = datetime.now()
        schema = GenomeSchema(role="test")
        after = datetime.now()

        assert before <= schema.discovered_at <= after

    def test_genome_schema_default_discovery_prompt(self):
        """GenomeSchema should have empty discovery_prompt by default."""
        schema = GenomeSchema(role="test")

        assert schema.discovery_prompt == ""

    def test_genome_schema_default_version(self):
        """GenomeSchema should have version 1 by default."""
        schema = GenomeSchema(role="test")

        assert schema.version == 1

    def test_genome_schema_with_traits(self):
        """GenomeSchema should accept traits dict."""
        traits = {
            "speed": GenomeTrait(name="speed", description="Work speed"),
            "accuracy": GenomeTrait(name="accuracy", description="Work accuracy"),
        }
        schema = GenomeSchema(role="worker", traits=traits)

        assert "speed" in schema.traits
        assert "accuracy" in schema.traits
        assert schema.traits["speed"].description == "Work speed"

    def test_genome_schema_with_discovery_prompt(self):
        """GenomeSchema should accept discovery_prompt."""
        schema = GenomeSchema(role="worker", discovery_prompt="Discover traits for a worker")

        assert schema.discovery_prompt == "Discover traits for a worker"

    def test_genome_schema_with_version(self):
        """GenomeSchema should accept version."""
        schema = GenomeSchema(role="worker", version=3)

        assert schema.version == 3

    def test_genome_schema_traits_is_mutable(self):
        """GenomeSchema traits should be mutable."""
        schema = GenomeSchema(role="worker")
        new_trait = GenomeTrait(name="endurance", description="Stamina")

        schema.traits["endurance"] = new_trait

        assert schema.traits["endurance"].description == "Stamina"


class TestExternalInput:
    """Tests for ExternalInput dataclass."""

    def test_external_input_creation(self):
        """ExternalInput should instantiate with required fields."""
        ext_input = ExternalInput(name="customers", target_agent_id="alex", rate=0.1)

        assert ext_input.name == "customers"
        assert ext_input.target_agent_id == "alex"
        assert ext_input.rate == 0.1

    def test_external_input_default_variance(self):
        """ExternalInput should have zero variance by default."""
        ext_input = ExternalInput(name="test", target_agent_id="a1", rate=1.0)

        assert ext_input.variance == 0.0

    def test_external_input_default_particle_type(self):
        """ExternalInput should have empty particle_type by default."""
        ext_input = ExternalInput(name="test", target_agent_id="a1", rate=1.0)

        assert ext_input.particle_type == ""

    def test_external_input_default_payload_generator(self):
        """ExternalInput should have default payload generator returning empty dict."""
        ext_input = ExternalInput(name="test", target_agent_id="a1", rate=1.0)

        assert callable(ext_input.payload_generator)
        assert ext_input.payload_generator() == {}

    def test_external_input_default_schedule(self):
        """ExternalInput should have default schedule returning 1.0."""
        ext_input = ExternalInput(name="test", target_agent_id="a1", rate=1.0)

        assert callable(ext_input.schedule)
        assert ext_input.schedule(0) == 1.0
        assert ext_input.schedule(100) == 1.0
        assert ext_input.schedule(1000) == 1.0

    def test_external_input_with_variance(self):
        """ExternalInput should accept variance."""
        ext_input = ExternalInput(name="test", target_agent_id="a1", rate=1.0, variance=0.3)

        assert ext_input.variance == 0.3

    def test_external_input_with_particle_type(self):
        """ExternalInput should accept particle_type."""
        ext_input = ExternalInput(
            name="orders", target_agent_id="alex", rate=0.1, particle_type="customer_order"
        )

        assert ext_input.particle_type == "customer_order"

    def test_external_input_with_payload_generator(self):
        """ExternalInput should accept payload_generator."""

        def gen_payload():
            return {"item": "BLT", "size": "large"}

        ext_input = ExternalInput(
            name="orders", target_agent_id="alex", rate=0.1, payload_generator=gen_payload
        )

        payload = ext_input.payload_generator()
        assert payload == {"item": "BLT", "size": "large"}

    def test_external_input_with_schedule(self):
        """ExternalInput should accept schedule function."""

        def rush_schedule(tick):
            if 200 <= tick <= 400:
                return 2.0
            return 1.0

        ext_input = ExternalInput(
            name="orders", target_agent_id="alex", rate=0.1, schedule=rush_schedule
        )

        assert ext_input.schedule(100) == 1.0
        assert ext_input.schedule(300) == 2.0
        assert ext_input.schedule(500) == 1.0

    def test_external_input_rate_zero(self):
        """ExternalInput can have zero rate."""
        ext_input = ExternalInput(name="test", target_agent_id="a1", rate=0.0)

        assert ext_input.rate == 0.0


class TestWorld:
    """Tests for World dataclass."""

    def test_world_creation_empty(self):
        """World should instantiate with no arguments."""
        world = World()

        assert isinstance(world, World)

    def test_world_default_agents(self):
        """World should have empty agents dict by default."""
        world = World()

        assert world.agents == {}

    def test_world_default_links(self):
        """World should have empty links dict by default."""
        world = World()

        assert world.links == {}

    def test_world_default_particles(self):
        """World should have empty particles dict by default."""
        world = World()

        assert world.particles == {}

    def test_world_default_labels(self):
        """World should have empty labels dict by default."""
        world = World()

        assert world.labels == {}

    def test_world_default_schemas(self):
        """World should have empty schemas dict by default."""
        world = World()

        assert world.schemas == {}

    def test_world_default_tick(self):
        """World should have tick=0 by default."""
        world = World()

        assert world.tick == 0

    def test_world_default_time(self):
        """World should have time=0.0 by default."""
        world = World()

        assert world.time == 0.0

    def test_world_default_speed(self):
        """World should have speed=1.0 by default."""
        world = World()

        assert world.speed == 1.0

    def test_world_default_external_inputs(self):
        """World should have empty external_inputs list by default."""
        world = World()

        assert world.external_inputs == []

    def test_world_with_agents(self):
        """World should hold agents dict."""
        agent = Agent(id="a1", name="Test", role="test")
        world = World(agents={"a1": agent})

        assert "a1" in world.agents
        assert world.agents["a1"].name == "Test"

    def test_world_with_links(self):
        """World should hold links dict."""
        link = Link(id="l1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)
        world = World(links={"l1": link})

        assert "l1" in world.links
        assert world.links["l1"].link_type == LinkType.PEER

    def test_world_with_particles(self):
        """World should hold particles dict."""
        particle = Particle(id="p1", particle_type="order")
        world = World(particles={"p1": particle})

        assert "p1" in world.particles
        assert world.particles["p1"].particle_type == "order"

    def test_world_with_labels(self):
        """World should hold labels dict."""
        label = Label(name="Kitchen")
        world = World(labels={"Kitchen": label})

        assert "Kitchen" in world.labels

    def test_world_with_schemas(self):
        """World should hold schemas dict."""
        schema = GenomeSchema(role="worker")
        world = World(schemas={"worker": schema})

        assert "worker" in world.schemas

    def test_world_with_tick(self):
        """World should accept initial tick."""
        world = World(tick=100)

        assert world.tick == 100

    def test_world_with_time(self):
        """World should accept initial time."""
        world = World(time=50.5)

        assert world.time == 50.5

    def test_world_with_speed(self):
        """World should accept speed."""
        world = World(speed=2.0)

        assert world.speed == 2.0

    def test_world_with_external_inputs(self):
        """World should hold external_inputs list."""
        ext_input = ExternalInput(name="test", target_agent_id="a1", rate=0.1)
        world = World(external_inputs=[ext_input])

        assert len(world.external_inputs) == 1
        assert world.external_inputs[0].name == "test"

    def test_world_agents_is_mutable(self):
        """World agents dict should be mutable."""
        world = World()
        agent = Agent(id="a1", name="New Agent", role="test")

        world.agents["a1"] = agent

        assert "a1" in world.agents

    def test_world_links_is_mutable(self):
        """World links dict should be mutable."""
        world = World()
        link = Link(id="l1", source_id="a1", dest_id="a2", link_type=LinkType.PEER)

        world.links["l1"] = link

        assert "l1" in world.links

    def test_world_particles_is_mutable(self):
        """World particles dict should be mutable."""
        world = World()
        particle = Particle(id="p1", particle_type="test")

        world.particles["p1"] = particle

        assert "p1" in world.particles

    def test_world_labels_is_mutable(self):
        """World labels dict should be mutable."""
        world = World()
        label = Label(name="NewLabel")

        world.labels["NewLabel"] = label

        assert "NewLabel" in world.labels

    def test_world_schemas_is_mutable(self):
        """World schemas dict should be mutable."""
        world = World()
        schema = GenomeSchema(role="new_role")

        world.schemas["new_role"] = schema

        assert "new_role" in world.schemas

    def test_world_external_inputs_is_mutable(self):
        """World external_inputs list should be mutable."""
        world = World()
        ext_input = ExternalInput(name="new", target_agent_id="a1", rate=0.1)

        world.external_inputs.append(ext_input)

        assert len(world.external_inputs) == 1

    def test_world_tick_is_mutable(self):
        """World tick should be mutable."""
        world = World()

        world.tick = 42

        assert world.tick == 42

    def test_world_time_is_mutable(self):
        """World time should be mutable."""
        world = World()

        world.time = 123.45

        assert world.time == 123.45

    def test_multiple_worlds_have_independent_containers(self):
        """Different worlds should not share container instances."""
        world1 = World()
        world2 = World()

        world1.agents["a1"] = Agent(id="a1", name="Agent 1", role="test")
        world1.particles["p1"] = Particle(id="p1", particle_type="test")

        assert "a1" not in world2.agents
        assert "p1" not in world2.particles

    def test_world_holds_complete_simulation_state(self):
        """World should be able to hold complete simulation state."""
        # Create agents
        maria = Agent(id="maria", name="Maria", role="owner")
        tom = Agent(id="tom", name="Tom", role="sandwich_maker")

        # Create link
        link = Link(
            id="maria_to_tom", source_id="maria", dest_id="tom", link_type=LinkType.HIERARCHICAL
        )

        # Create label
        kitchen = Label(name="Kitchen", context=LabelContext(resources=["grill"], norms=["FIFO"]))

        # Create schema
        schema = GenomeSchema(role="sandwich_maker")

        # Create external input
        ext_input = ExternalInput(name="customers", target_agent_id="tom", rate=0.1)

        # Create particle
        particle = Particle(
            id="p1", particle_type="order", source_id="maria", dest_id="tom", link_id="maria_to_tom"
        )

        # Assemble world
        world = World(
            agents={"maria": maria, "tom": tom},
            links={"maria_to_tom": link},
            labels={"Kitchen": kitchen},
            schemas={"sandwich_maker": schema},
            external_inputs=[ext_input],
            particles={"p1": particle},
            tick=10,
            time=10.0,
            speed=1.0,
        )

        # Verify all state
        assert len(world.agents) == 2
        assert len(world.links) == 1
        assert len(world.labels) == 1
        assert len(world.schemas) == 1
        assert len(world.external_inputs) == 1
        assert len(world.particles) == 1
        assert world.tick == 10
        assert world.time == 10.0

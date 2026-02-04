"""Tests for the frame projection module."""

import math

from loopengine.model.agent import Agent, Phase
from loopengine.model.link import Link, LinkType
from loopengine.model.particle import Particle
from loopengine.model.world import World
from loopengine.projection import (
    AgentVisual,
    Frame,
    LabelRegionVisual,
    LinkVisual,
    ParticleVisual,
    compute_breathing_radius,
    project,
)
from loopengine.projection.projector import (
    _bezier_point,
    _compute_bezier_control_points,
    _convex_hull,
    _expand_hull,
    _interpolate_particle_position,
    _project_agent,
    _project_label_regions,
    _project_link,
    _project_particle,
)


def make_agent(
    agent_id: str,
    name: str = "",
    role: str = "tester",
    x: float = 0.0,
    y: float = 0.0,
    labels: set[str] | None = None,
    input_buffer: list[Particle] | None = None,
    phase: Phase = Phase.SENSE,
) -> Agent:
    """Create a test agent."""
    return Agent(
        id=agent_id,
        name=name or f"Agent {agent_id}",
        role=role,
        x=x,
        y=y,
        labels=labels if labels is not None else set(),
        input_buffer=input_buffer if input_buffer is not None else [],
        loop_phase=phase,
    )


def make_link(
    link_id: str,
    source_id: str,
    dest_id: str,
    link_type: LinkType = LinkType.PEER,
) -> Link:
    """Create a test link."""
    return Link(
        id=link_id,
        source_id=source_id,
        dest_id=dest_id,
        link_type=link_type,
    )


def make_particle(
    particle_id: str,
    particle_type: str = "order_ticket",
    source_id: str = "",
    dest_id: str = "",
    link_id: str = "",
    progress: float = 0.5,
) -> Particle:
    """Create a test particle."""
    return Particle(
        id=particle_id,
        particle_type=particle_type,
        source_id=source_id,
        dest_id=dest_id,
        link_id=link_id,
        progress=progress,
    )


def make_world() -> World:
    """Create an empty test world."""
    return World()


class TestAgentVisualDataclass:
    """Tests for AgentVisual dataclass."""

    def test_agent_visual_has_required_fields(self):
        """AgentVisual should have all required fields."""
        visual = AgentVisual(
            id="agent1",
            name="Test Agent",
            role="tester",
            x=100.0,
            y=200.0,
        )

        assert visual.id == "agent1"
        assert visual.name == "Test Agent"
        assert visual.role == "tester"
        assert visual.x == 100.0
        assert visual.y == 200.0

    def test_agent_visual_defaults(self):
        """AgentVisual should have sensible defaults."""
        visual = AgentVisual(id="a1", name="Test", role="r", x=0, y=0)

        assert visual.radius == 20.0
        assert visual.breathing_phase == 0.0
        assert visual.breathing_rate == 0.05
        assert visual.glow_intensity == 0.0
        assert visual.ooda_phase == "sense"
        assert visual.labels == []


class TestLinkVisualDataclass:
    """Tests for LinkVisual dataclass."""

    def test_link_visual_has_required_fields(self):
        """LinkVisual should have all required fields."""
        visual = LinkVisual(
            id="link1",
            source_id="a1",
            dest_id="a2",
            link_type="peer",
        )

        assert visual.id == "link1"
        assert visual.source_id == "a1"
        assert visual.dest_id == "a2"
        assert visual.link_type == "peer"

    def test_link_visual_defaults(self):
        """LinkVisual should have sensible defaults."""
        visual = LinkVisual(id="l1", source_id="a1", dest_id="a2", link_type="peer")

        assert visual.thickness == 2.0
        assert visual.sway_phase == 0.0
        assert visual.control_points == []


class TestParticleVisualDataclass:
    """Tests for ParticleVisual dataclass."""

    def test_particle_visual_has_required_fields(self):
        """ParticleVisual should have all required fields."""
        visual = ParticleVisual(
            id="p1",
            particle_type="order_ticket",
            x=50.0,
            y=75.0,
        )

        assert visual.id == "p1"
        assert visual.particle_type == "order_ticket"
        assert visual.x == 50.0
        assert visual.y == 75.0

    def test_particle_visual_defaults(self):
        """ParticleVisual should have sensible defaults."""
        visual = ParticleVisual(id="p1", particle_type="test", x=0, y=0)

        assert visual.size == 6.0
        assert visual.trail == []


class TestLabelRegionVisualDataclass:
    """Tests for LabelRegionVisual dataclass."""

    def test_label_region_visual_has_required_fields(self):
        """LabelRegionVisual should have required fields."""
        visual = LabelRegionVisual(
            name="team_a",
            hull_points=[(0, 0), (100, 0), (50, 100)],
        )

        assert visual.name == "team_a"
        assert len(visual.hull_points) == 3


class TestFrameDataclass:
    """Tests for Frame dataclass."""

    def test_frame_has_required_fields(self):
        """Frame should have tick and time."""
        frame = Frame(tick=100, time=10.0)

        assert frame.tick == 100
        assert frame.time == 10.0

    def test_frame_defaults_to_empty_lists(self):
        """Frame collections should default to empty."""
        frame = Frame(tick=0, time=0.0)

        assert frame.agents == []
        assert frame.links == []
        assert frame.particles == []
        assert frame.label_regions == []


class TestProjectAgent:
    """Tests for _project_agent function."""

    def test_project_agent_copies_basic_fields(self):
        """Projected agent should have correct id, name, role, position."""
        agent = make_agent("a1", name="Tom", role="sandwich_maker", x=100.0, y=200.0)

        visual = _project_agent(agent, tick=0)

        assert visual.id == "a1"
        assert visual.name == "Tom"
        assert visual.role == "sandwich_maker"
        assert visual.x == 100.0
        assert visual.y == 200.0

    def test_project_agent_breathing_phase_varies_with_tick(self):
        """Breathing phase should change based on tick."""
        agent = make_agent("a1")

        visual_t0 = _project_agent(agent, tick=0)
        visual_t10 = _project_agent(agent, tick=10)
        visual_t100 = _project_agent(agent, tick=100)

        # Phases should be different
        assert visual_t0.breathing_phase != visual_t10.breathing_phase
        assert visual_t10.breathing_phase != visual_t100.breathing_phase

    def test_project_agent_glow_reflects_input_buffer(self):
        """Glow intensity should increase with input buffer depth."""
        empty_buffer_agent = make_agent("a1", input_buffer=[])
        full_buffer_agent = make_agent(
            "a2",
            input_buffer=[make_particle(f"p{i}") for i in range(5)],
        )

        visual_empty = _project_agent(empty_buffer_agent, tick=0)
        visual_full = _project_agent(full_buffer_agent, tick=0)

        assert visual_empty.glow_intensity == 0.0
        assert visual_full.glow_intensity > 0.0
        assert visual_full.glow_intensity == 1.0  # 5 * 0.2 = 1.0 (capped)

    def test_project_agent_glow_caps_at_one(self):
        """Glow intensity should cap at 1.0."""
        agent = make_agent(
            "a1",
            input_buffer=[make_particle(f"p{i}") for i in range(20)],
        )

        visual = _project_agent(agent, tick=0)

        assert visual.glow_intensity == 1.0

    def test_project_agent_ooda_phase_matches(self):
        """OODA phase should reflect agent's current phase."""
        sense_agent = make_agent("a1", phase=Phase.SENSE)
        act_agent = make_agent("a2", phase=Phase.ACT)

        visual_sense = _project_agent(sense_agent, tick=0)
        visual_act = _project_agent(act_agent, tick=0)

        assert visual_sense.ooda_phase == "sense"
        assert visual_act.ooda_phase == "act"

    def test_project_agent_includes_labels(self):
        """Projected agent should include labels."""
        agent = make_agent("a1", labels={"kitchen", "morning_shift"})

        visual = _project_agent(agent, tick=0)

        assert set(visual.labels) == {"kitchen", "morning_shift"}


class TestProjectLink:
    """Tests for _project_link function."""

    def test_project_link_copies_basic_fields(self):
        """Projected link should have correct ids and type."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=100)

        link = make_link("l1", "a1", "a2", LinkType.HIERARCHICAL)
        world.links["l1"] = link

        visual = _project_link(link, world)

        assert visual.id == "l1"
        assert visual.source_id == "a1"
        assert visual.dest_id == "a2"
        assert visual.link_type == "hierarchical"

    def test_project_link_returns_none_for_missing_agents(self):
        """Should return None if source or dest agent missing."""
        world = make_world()
        world.agents["a1"] = make_agent("a1")
        # a2 is missing

        link = make_link("l1", "a1", "a2")

        visual = _project_link(link, world)

        assert visual is None

    def test_project_link_has_control_points(self):
        """Projected link should have bezier control points."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=0)

        link = make_link("l1", "a1", "a2")
        world.links["l1"] = link

        visual = _project_link(link, world)

        assert len(visual.control_points) == 4
        # First point should be source position
        assert visual.control_points[0] == (0.0, 0.0)
        # Last point should be dest position
        assert visual.control_points[3] == (100.0, 0.0)


class TestProjectParticle:
    """Tests for _project_particle function."""

    def test_project_particle_interpolates_position(self):
        """Particle position should be interpolated along link."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=0)

        link = make_link("l1", "a1", "a2")
        world.links["l1"] = link

        # Particle at 50% progress
        particle = make_particle("p1", source_id="a1", dest_id="a2", link_id="l1", progress=0.5)
        world.particles["p1"] = particle

        visual = _project_particle(particle, world)

        assert visual is not None
        # Should be approximately halfway (bezier curve may not be exactly linear)
        assert 30.0 < visual.x < 70.0

    def test_project_particle_at_source(self):
        """Particle at 0% progress should be near source."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=0)

        link = make_link("l1", "a1", "a2")
        world.links["l1"] = link

        particle = make_particle("p1", source_id="a1", dest_id="a2", link_id="l1", progress=0.0)
        world.particles["p1"] = particle

        visual = _project_particle(particle, world)

        assert visual.x == 0.0
        assert visual.y == 0.0

    def test_project_particle_at_dest(self):
        """Particle at 100% progress should be at dest."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=0)

        link = make_link("l1", "a1", "a2")
        world.links["l1"] = link

        particle = make_particle("p1", source_id="a1", dest_id="a2", link_id="l1", progress=1.0)
        world.particles["p1"] = particle

        visual = _project_particle(particle, world)

        assert visual.x == 100.0
        assert visual.y == 0.0

    def test_project_particle_copies_type(self):
        """Particle type should be preserved."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=0)

        link = make_link("l1", "a1", "a2")
        world.links["l1"] = link

        particle = make_particle(
            "p1", particle_type="sandwich", source_id="a1", dest_id="a2", link_id="l1"
        )
        world.particles["p1"] = particle

        visual = _project_particle(particle, world)

        assert visual.particle_type == "sandwich"


class TestBezierControlPoints:
    """Tests for _compute_bezier_control_points function."""

    def test_control_points_start_and_end(self):
        """First and last control points should be source and dest."""
        points = _compute_bezier_control_points(0, 0, 100, 100)

        assert points[0] == (0, 0)
        assert points[3] == (100, 100)

    def test_control_points_count(self):
        """Should return 4 control points."""
        points = _compute_bezier_control_points(0, 0, 100, 0)

        assert len(points) == 4

    def test_control_points_degenerate_case(self):
        """Same start and end should not crash."""
        points = _compute_bezier_control_points(50, 50, 50, 50)

        assert len(points) == 4
        assert points[0] == (50, 50)
        assert points[3] == (50, 50)


class TestBezierPoint:
    """Tests for _bezier_point function."""

    def test_bezier_at_t0(self):
        """t=0 should return first point."""
        points = [(0, 0), (25, 50), (75, 50), (100, 0)]

        x, y = _bezier_point(points, 0.0)

        assert x == 0.0
        assert y == 0.0

    def test_bezier_at_t1(self):
        """t=1 should return last point."""
        points = [(0, 0), (25, 50), (75, 50), (100, 0)]

        x, y = _bezier_point(points, 1.0)

        assert x == 100.0
        assert y == 0.0

    def test_bezier_at_t_half(self):
        """t=0.5 should be somewhere between start and end."""
        points = [(0, 0), (25, 50), (75, 50), (100, 0)]

        x, y = _bezier_point(points, 0.5)

        # Should be roughly in the middle
        assert 40 < x < 60
        assert y > 0  # Curve bows outward


class TestConvexHull:
    """Tests for _convex_hull function."""

    def test_hull_of_triangle(self):
        """Hull of 3 non-collinear points is those points."""
        points = [(0, 0), (100, 0), (50, 100)]

        hull = _convex_hull(points)

        assert len(hull) == 3
        assert set(hull) == set(points)

    def test_hull_excludes_interior_point(self):
        """Hull should not include interior points."""
        points = [(0, 0), (100, 0), (0, 100), (100, 100), (50, 50)]  # Interior point

        hull = _convex_hull(points)

        assert len(hull) == 4
        assert (50, 50) not in hull

    def test_hull_with_two_points(self):
        """Hull of 2 points is those 2 points."""
        points = [(0, 0), (100, 100)]

        hull = _convex_hull(points)

        assert len(hull) == 2


class TestExpandHull:
    """Tests for _expand_hull function."""

    def test_expand_hull_increases_size(self):
        """Expanded hull should be larger."""
        hull = [(0, 0), (100, 0), (50, 100)]
        centroid_x = 50
        centroid_y = 33.33

        expanded = _expand_hull(hull, padding=10.0)

        # Each point should be farther from centroid
        for orig, exp in zip(hull, expanded, strict=True):
            orig_dist = math.sqrt((orig[0] - centroid_x) ** 2 + (orig[1] - centroid_y) ** 2)
            exp_dist = math.sqrt((exp[0] - centroid_x) ** 2 + (exp[1] - centroid_y) ** 2)
            assert exp_dist > orig_dist


class TestProjectLabelRegions:
    """Tests for _project_label_regions function."""

    def test_label_region_for_shared_label(self):
        """Agents sharing a label should create a region."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0, labels={"team"})
        world.agents["a2"] = make_agent("a2", x=100, y=0, labels={"team"})
        world.agents["a3"] = make_agent("a3", x=50, y=100, labels={"team"})

        regions = _project_label_regions(world)

        assert len(regions) == 1
        assert regions[0].name == "team"
        assert len(regions[0].hull_points) >= 3

    def test_no_region_for_single_agent(self):
        """Single agent with label should not create region."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", labels={"solo"})

        regions = _project_label_regions(world)

        assert len(regions) == 0

    def test_multiple_regions_for_multiple_labels(self):
        """Different labels should create different regions."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0, labels={"team_a"})
        world.agents["a2"] = make_agent("a2", x=50, y=0, labels={"team_a"})
        world.agents["a3"] = make_agent("a3", x=200, y=0, labels={"team_b"})
        world.agents["a4"] = make_agent("a4", x=250, y=0, labels={"team_b"})

        regions = _project_label_regions(world)

        assert len(regions) == 2
        region_names = {r.name for r in regions}
        assert region_names == {"team_a", "team_b"}


class TestProject:
    """Tests for the main project() function."""

    def test_project_returns_frame(self):
        """project() should return a Frame."""
        world = make_world()
        world.tick = 100
        world.time = 10.0

        frame = project(world)

        assert isinstance(frame, Frame)
        assert frame.tick == 100
        assert frame.time == 10.0

    def test_project_includes_all_agents(self):
        """Frame should contain visual for each agent."""
        world = make_world()
        world.agents["a1"] = make_agent("a1")
        world.agents["a2"] = make_agent("a2")
        world.agents["a3"] = make_agent("a3")

        frame = project(world)

        assert len(frame.agents) == 3
        agent_ids = {a.id for a in frame.agents}
        assert agent_ids == {"a1", "a2", "a3"}

    def test_project_includes_all_links(self):
        """Frame should contain visual for each valid link."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=0)
        world.links["l1"] = make_link("l1", "a1", "a2")
        world.links["l2"] = make_link("l2", "a2", "a1")

        frame = project(world)

        assert len(frame.links) == 2

    def test_project_includes_alive_particles(self):
        """Frame should contain visual for alive particles."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=0)
        world.links["l1"] = make_link("l1", "a1", "a2")

        alive_particle = make_particle("p1", source_id="a1", dest_id="a2", link_id="l1")
        alive_particle.alive = True

        dead_particle = make_particle("p2", source_id="a1", dest_id="a2", link_id="l1")
        dead_particle.alive = False

        world.particles["p1"] = alive_particle
        world.particles["p2"] = dead_particle

        frame = project(world)

        assert len(frame.particles) == 1
        assert frame.particles[0].id == "p1"

    def test_project_includes_label_regions(self):
        """Frame should contain label regions."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0, labels={"kitchen"})
        world.agents["a2"] = make_agent("a2", x=100, y=0, labels={"kitchen"})

        frame = project(world)

        assert len(frame.label_regions) == 1
        assert frame.label_regions[0].name == "kitchen"


class TestComputeBreathingRadius:
    """Tests for compute_breathing_radius function."""

    def test_breathing_radius_at_phase_zero(self):
        """At phase 0, sin(0)=0, radius should be base."""
        radius = compute_breathing_radius(20.0, 0.0)

        assert radius == 20.0

    def test_breathing_radius_at_phase_pi_half(self):
        """At phase π/2, sin(π/2)=1, radius should be base * 1.05."""
        radius = compute_breathing_radius(20.0, math.pi / 2)

        assert abs(radius - 21.0) < 0.001  # 20 * 1.05

    def test_breathing_radius_at_phase_pi(self):
        """At phase π, sin(π)≈0, radius should be base."""
        radius = compute_breathing_radius(20.0, math.pi)

        assert abs(radius - 20.0) < 0.001

    def test_breathing_radius_at_phase_3pi_half(self):
        """At phase 3π/2, sin(3π/2)=-1, radius should be base * 0.95."""
        radius = compute_breathing_radius(20.0, 3 * math.pi / 2)

        assert abs(radius - 19.0) < 0.001  # 20 * 0.95


class TestInterpolateParticlePosition:
    """Tests for _interpolate_particle_position function."""

    def test_interpolate_with_link(self):
        """Particle on a link should interpolate along bezier."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=0)

        link = make_link("l1", "a1", "a2")
        world.links["l1"] = link

        particle = make_particle("p1", source_id="a1", dest_id="a2", link_id="l1", progress=0.5)

        x, y = _interpolate_particle_position(particle, world)

        assert x is not None
        assert y is not None
        assert 30 < x < 70  # Approximately middle

    def test_interpolate_without_link_uses_agents(self):
        """Particle without link should interpolate between agents directly."""
        world = make_world()
        world.agents["a1"] = make_agent("a1", x=0, y=0)
        world.agents["a2"] = make_agent("a2", x=100, y=0)

        particle = make_particle("p1", source_id="a1", dest_id="a2", link_id="", progress=0.5)

        x, y = _interpolate_particle_position(particle, world)

        assert x == 50.0  # Exactly halfway (linear)
        assert y == 0.0

    def test_interpolate_missing_agents_returns_none(self):
        """Missing agents should return None."""
        world = make_world()

        particle = make_particle("p1", source_id="missing1", dest_id="missing2", progress=0.5)

        x, y = _interpolate_particle_position(particle, world)

        assert x is None
        assert y is None

"""Tests for the Software Team corpus."""

from loopengine.corpora.software_team import (
    BUG_SEVERITIES,
    DESIGN_DELIVERABLES,
    FEATURE_TYPES,
    create_agents,
    create_external_inputs,
    create_labels,
    create_links,
    create_software_team_world,
    create_world,
    designer_policy,
    dev_policy,
    generate_bug_report_payload,
    generate_feature_request_payload,
    pm_policy,
    sprint_schedule,
)
from loopengine.model import LinkType, Particle

# ============================================================================
# LINK TESTS
# ============================================================================


def test_links_count() -> None:
    """Verify 10 links are created."""
    links = create_links()
    assert len(links) == 10


def test_links_ids() -> None:
    """Verify all link IDs match expected naming."""
    links = create_links()
    expected_ids = {
        "pm_to_dev1",
        "pm_to_dev2",
        "dev1_to_pm",
        "dev2_to_pm",
        "dev1_to_dev2",
        "dev2_to_dev1",
        "designer_to_dev1",
        "designer_to_dev2",
        "dev1_to_designer",
        "dev2_to_designer",
    }
    assert set(links.keys()) == expected_ids


def test_links_are_directional() -> None:
    """Verify links are directional (dev1_to_dev2 separate from dev2_to_dev1)."""
    links = create_links()

    dev1_to_dev2 = links["dev1_to_dev2"]
    dev2_to_dev1 = links["dev2_to_dev1"]

    assert dev1_to_dev2.source_id == "dev1"
    assert dev1_to_dev2.dest_id == "dev2"
    assert dev2_to_dev1.source_id == "dev2"
    assert dev2_to_dev1.dest_id == "dev1"


def test_pm_to_dev1_hierarchical() -> None:
    """Verify pm_to_dev1 has HIERARCHICAL type with authority_scope."""
    links = create_links()
    link = links["pm_to_dev1"]

    assert link.link_type == LinkType.HIERARCHICAL
    assert link.source_id == "pm"
    assert link.dest_id == "dev1"
    assert "authority_scope" in link.properties
    assert "task_assignment" in link.properties["authority_scope"]
    assert "autonomy_granted" in link.properties
    assert "fitness_definition" in link.properties
    assert "flow_types" in link.properties


def test_pm_to_dev2_hierarchical() -> None:
    """Verify pm_to_dev2 has HIERARCHICAL type with authority_scope."""
    links = create_links()
    link = links["pm_to_dev2"]

    assert link.link_type == LinkType.HIERARCHICAL
    assert link.source_id == "pm"
    assert link.dest_id == "dev2"
    assert "authority_scope" in link.properties


def test_dev1_to_pm_hierarchical_upward() -> None:
    """Verify dev1_to_pm is HIERARCHICAL upward with flow_types."""
    links = create_links()
    link = links["dev1_to_pm"]

    assert link.link_type == LinkType.HIERARCHICAL
    assert link.source_id == "dev1"
    assert link.dest_id == "pm"
    assert "flow_types" in link.properties
    assert "status_update" in link.properties["flow_types"]
    assert "completion_notice" in link.properties["flow_types"]


def test_dev2_to_pm_hierarchical_upward() -> None:
    """Verify dev2_to_pm is HIERARCHICAL upward with flow_types."""
    links = create_links()
    link = links["dev2_to_pm"]

    assert link.link_type == LinkType.HIERARCHICAL
    assert link.source_id == "dev2"
    assert link.dest_id == "pm"
    assert "flow_types" in link.properties


def test_dev1_to_dev2_peer() -> None:
    """Verify dev1_to_dev2 has PEER type with flow_types and bandwidth."""
    links = create_links()
    link = links["dev1_to_dev2"]

    assert link.link_type == LinkType.PEER
    assert link.source_id == "dev1"
    assert link.dest_id == "dev2"
    assert "flow_types" in link.properties
    assert "code_review_request" in link.properties["flow_types"]
    assert "bandwidth" in link.properties


def test_dev2_to_dev1_peer() -> None:
    """Verify dev2_to_dev1 has PEER type."""
    links = create_links()
    link = links["dev2_to_dev1"]

    assert link.link_type == LinkType.PEER
    assert link.source_id == "dev2"
    assert link.dest_id == "dev1"


def test_designer_to_dev1_service() -> None:
    """Verify designer_to_dev1 has SERVICE type."""
    links = create_links()
    link = links["designer_to_dev1"]

    assert link.link_type == LinkType.SERVICE
    assert link.source_id == "designer"
    assert link.dest_id == "dev1"
    assert "flow_types" in link.properties
    assert "design_spec" in link.properties["flow_types"]


def test_designer_to_dev2_service() -> None:
    """Verify designer_to_dev2 has SERVICE type."""
    links = create_links()
    link = links["designer_to_dev2"]

    assert link.link_type == LinkType.SERVICE
    assert link.source_id == "designer"
    assert link.dest_id == "dev2"


def test_dev1_to_designer_service() -> None:
    """Verify dev1_to_designer has SERVICE type."""
    links = create_links()
    link = links["dev1_to_designer"]

    assert link.link_type == LinkType.SERVICE
    assert link.source_id == "dev1"
    assert link.dest_id == "designer"
    assert "flow_types" in link.properties
    assert "design_request" in link.properties["flow_types"]


def test_dev2_to_designer_service() -> None:
    """Verify dev2_to_designer has SERVICE type."""
    links = create_links()
    link = links["dev2_to_designer"]

    assert link.link_type == LinkType.SERVICE
    assert link.source_id == "dev2"
    assert link.dest_id == "designer"


def test_create_world_has_links() -> None:
    """Verify create_world populates world.links."""
    world = create_world()
    assert len(world.links) == 10
    assert "pm_to_dev1" in world.links


# ============================================================================
# LABEL TESTS
# ============================================================================


def test_labels_count() -> None:
    """Verify 5 labels are created."""
    labels = create_labels()
    assert len(labels) == 5


def test_labels_names() -> None:
    """Verify all label names match expected."""
    labels = create_labels()
    expected_names = {"SoftwareTeam", "Engineering", "Design", "Management", "Codebase"}
    assert set(labels.keys()) == expected_names


def test_softwareteam_label_constraints() -> None:
    """Verify SoftwareTeam label has development constraints."""
    labels = create_labels()
    label = labels["SoftwareTeam"]

    assert label.name == "SoftwareTeam"
    assert "code_review_required" in label.context.constraints
    assert "documentation_standards" in label.context.constraints
    assert "test_coverage_80" in label.context.constraints


def test_softwareteam_label_resources() -> None:
    """Verify SoftwareTeam label has correct resources."""
    labels = create_labels()
    label = labels["SoftwareTeam"]

    assert "git_repository" in label.context.resources
    assert "ci_pipeline" in label.context.resources
    assert "issue_tracker" in label.context.resources


def test_softwareteam_label_norms() -> None:
    """Verify SoftwareTeam label has correct norms."""
    labels = create_labels()
    label = labels["SoftwareTeam"]

    assert "daily_standup" in label.context.norms
    assert "sprint_planning" in label.context.norms
    assert "retrospective" in label.context.norms


def test_engineering_label() -> None:
    """Verify Engineering label has development context."""
    labels = create_labels()
    label = labels["Engineering"]

    assert label.name == "Engineering"
    assert "code_style_guide" in label.context.constraints
    assert "ide" in label.context.resources
    assert "code_review" in label.context.norms


def test_design_label() -> None:
    """Verify Design label has design context."""
    labels = create_labels()
    label = labels["Design"]

    assert label.name == "Design"
    assert "accessibility_standards" in label.context.constraints
    assert "figma" in label.context.resources
    assert "design_review" in label.context.norms


def test_management_label() -> None:
    """Verify Management label has PM context."""
    labels = create_labels()
    label = labels["Management"]

    assert label.name == "Management"
    assert "roadmap" in label.context.resources
    assert "backlog" in label.context.resources
    assert "prioritization" in label.context.norms


def test_codebase_label() -> None:
    """Verify Codebase label has repository context."""
    labels = create_labels()
    label = labels["Codebase"]

    assert label.name == "Codebase"
    assert "branch_naming" in label.context.constraints
    assert "main_branch" in label.context.resources


def test_create_world_has_labels() -> None:
    """Verify create_world populates world.labels."""
    world = create_world()
    assert len(world.labels) == 5
    assert "SoftwareTeam" in world.labels
    assert "Engineering" in world.labels


# ============================================================================
# EXTERNAL INPUT TESTS
# ============================================================================


def test_external_inputs_count() -> None:
    """Verify 2 external inputs are created (feature_requests, bug_reports)."""
    inputs = create_external_inputs()
    assert len(inputs) == 2


def test_feature_requests_external_input() -> None:
    """Verify feature_requests external input targeting PM with rate=0.03."""
    inputs = create_external_inputs()
    feature_requests = next(i for i in inputs if i.name == "feature_requests")

    assert feature_requests.name == "feature_requests"
    assert feature_requests.target_agent_id == "pm"
    assert feature_requests.rate == 0.03
    assert feature_requests.variance == 0.2
    assert feature_requests.particle_type == "feature_request"


def test_bug_reports_external_input() -> None:
    """Verify bug_reports external input targeting PM with rate=0.02."""
    inputs = create_external_inputs()
    bug_reports = next(i for i in inputs if i.name == "bug_reports")

    assert bug_reports.name == "bug_reports"
    assert bug_reports.target_agent_id == "pm"
    assert bug_reports.rate == 0.02
    assert bug_reports.variance == 0.3
    assert bug_reports.particle_type == "bug_report"


def test_sprint_schedule_start() -> None:
    """Verify schedule returns 1.5 at sprint start (ticks <= 100)."""
    assert sprint_schedule(0) == 1.5
    assert sprint_schedule(50) == 1.5
    assert sprint_schedule(100) == 1.5


def test_sprint_schedule_normal() -> None:
    """Verify schedule returns 1.0 during normal sprint (ticks 101-899)."""
    assert sprint_schedule(101) == 1.0
    assert sprint_schedule(500) == 1.0
    assert sprint_schedule(899) == 1.0


def test_sprint_schedule_end() -> None:
    """Verify schedule returns 0.5 at sprint end (ticks >= 900)."""
    assert sprint_schedule(900) == 0.5
    assert sprint_schedule(950) == 0.5
    assert sprint_schedule(1000) == 0.5


def test_feature_request_payload_structure() -> None:
    """Verify payload generator returns dict with feature_type, description, priority."""
    payload = generate_feature_request_payload()

    assert "feature_type" in payload
    assert "description" in payload
    assert "priority" in payload
    assert "estimated_effort" in payload


def test_feature_request_payload_values() -> None:
    """Verify feature_type is from valid list."""
    for _ in range(20):
        payload = generate_feature_request_payload()
        assert payload["feature_type"] in FEATURE_TYPES
        assert payload["priority"] in ["high", "medium", "low"]
        assert payload["estimated_effort"] in ["small", "medium", "large"]


def test_bug_report_payload_structure() -> None:
    """Verify payload generator returns dict with severity, component, title."""
    payload = generate_bug_report_payload()

    assert "severity" in payload
    assert "component" in payload
    assert "title" in payload
    assert "reproduced" in payload


def test_bug_report_payload_values() -> None:
    """Verify severity is from valid list."""
    for _ in range(20):
        payload = generate_bug_report_payload()
        assert payload["severity"] in BUG_SEVERITIES
        assert payload["component"] in FEATURE_TYPES


def test_external_input_has_callable_schedule() -> None:
    """Verify external input schedule is callable and returns rate multiplier."""
    inputs = create_external_inputs()
    feature_requests = next(i for i in inputs if i.name == "feature_requests")

    assert callable(feature_requests.schedule)
    assert feature_requests.schedule(50) == 1.5
    assert feature_requests.schedule(500) == 1.0


def test_external_input_has_callable_payload_generator() -> None:
    """Verify external input payload_generator is callable and returns dict."""
    inputs = create_external_inputs()
    feature_requests = next(i for i in inputs if i.name == "feature_requests")

    assert callable(feature_requests.payload_generator)
    payload = feature_requests.payload_generator()
    assert isinstance(payload, dict)


def test_create_world_has_external_inputs() -> None:
    """Verify create_world populates world.external_inputs."""
    world = create_world()
    assert len(world.external_inputs) == 2
    assert any(i.name == "feature_requests" for i in world.external_inputs)
    assert any(i.name == "bug_reports" for i in world.external_inputs)


# ============================================================================
# AGENT TESTS
# ============================================================================


def test_agents_count() -> None:
    """Verify 4 agents are created."""
    agents = create_agents()
    assert len(agents) == 4


def test_agents_ids() -> None:
    """Verify all agent IDs match expected naming."""
    agents = create_agents()
    expected_ids = {"pm", "dev1", "dev2", "designer"}
    assert set(agents.keys()) == expected_ids


def test_pm_agent_role() -> None:
    """Verify PM agent has product_manager role."""
    agents = create_agents()
    pm = agents["pm"]

    assert pm.id == "pm"
    assert pm.name == "PM"
    assert pm.role == "product_manager"


def test_pm_agent_loop_period() -> None:
    """Verify PM has loop_period=200 (slower, strategic)."""
    agents = create_agents()
    pm = agents["pm"]
    assert pm.loop_period == 200


def test_pm_agent_labels() -> None:
    """Verify PM has Management label."""
    agents = create_agents()
    pm = agents["pm"]

    assert "SoftwareTeam" in pm.labels
    assert "Management" in pm.labels


def test_pm_agent_genome() -> None:
    """Verify PM has correct initial genome values."""
    agents = create_agents()
    pm = agents["pm"]

    assert pm.genome["prioritization"] == 0.8
    assert pm.genome["stakeholder_management"] == 0.7
    assert pm.genome["team_coordination"] == 0.8
    assert pm.genome["decisiveness"] == 0.6
    assert pm.genome["communication"] == 0.8


def test_pm_agent_has_policy() -> None:
    """Verify PM has a working policy callable."""
    agents = create_agents()
    pm = agents["pm"]

    assert callable(pm.policy)
    result = pm.policy([], pm.genome, {})
    assert isinstance(result, list)


def test_dev1_agent_role() -> None:
    """Verify Dev1 agent has developer role."""
    agents = create_agents()
    dev1 = agents["dev1"]

    assert dev1.id == "dev1"
    assert dev1.name == "Dev1"
    assert dev1.role == "developer"


def test_dev1_agent_loop_period() -> None:
    """Verify Dev1 has loop_period=40 (medium, coding)."""
    agents = create_agents()
    dev1 = agents["dev1"]
    assert dev1.loop_period == 40


def test_dev1_agent_labels() -> None:
    """Verify Dev1 has Engineering and Codebase labels."""
    agents = create_agents()
    dev1 = agents["dev1"]

    assert "SoftwareTeam" in dev1.labels
    assert "Engineering" in dev1.labels
    assert "Codebase" in dev1.labels


def test_dev1_agent_genome() -> None:
    """Verify Dev1 has correct initial genome values."""
    agents = create_agents()
    dev1 = agents["dev1"]

    assert dev1.genome["coding_speed"] == 0.7
    assert dev1.genome["code_quality"] == 0.8
    assert dev1.genome["collaboration"] == 0.7
    assert dev1.genome["review_thoroughness"] == 0.7
    assert dev1.genome["debugging_skill"] == 0.6


def test_dev1_agent_has_policy() -> None:
    """Verify Dev1 has a working policy callable."""
    agents = create_agents()
    dev1 = agents["dev1"]

    assert callable(dev1.policy)
    result = dev1.policy([], dev1.genome, {"agent_id": "dev1"})
    assert isinstance(result, list)


def test_dev2_agent_role() -> None:
    """Verify Dev2 agent has developer role."""
    agents = create_agents()
    dev2 = agents["dev2"]

    assert dev2.id == "dev2"
    assert dev2.name == "Dev2"
    assert dev2.role == "developer"


def test_dev2_agent_loop_period() -> None:
    """Verify Dev2 has loop_period=40 (same as Dev1)."""
    agents = create_agents()
    dev2 = agents["dev2"]
    assert dev2.loop_period == 40


def test_dev2_agent_labels() -> None:
    """Verify Dev2 has Engineering and Codebase labels."""
    agents = create_agents()
    dev2 = agents["dev2"]

    assert "SoftwareTeam" in dev2.labels
    assert "Engineering" in dev2.labels
    assert "Codebase" in dev2.labels


def test_dev2_agent_genome() -> None:
    """Verify Dev2 has different genome values (higher quality, lower speed)."""
    agents = create_agents()
    dev2 = agents["dev2"]

    assert dev2.genome["coding_speed"] == 0.6
    assert dev2.genome["code_quality"] == 0.85
    assert dev2.genome["collaboration"] == 0.8
    assert dev2.genome["review_thoroughness"] == 0.8
    assert dev2.genome["debugging_skill"] == 0.7


def test_dev2_agent_has_policy() -> None:
    """Verify Dev2 has a working policy callable."""
    agents = create_agents()
    dev2 = agents["dev2"]

    assert callable(dev2.policy)
    result = dev2.policy([], dev2.genome, {"agent_id": "dev2"})
    assert isinstance(result, list)


def test_designer_agent_role() -> None:
    """Verify Designer agent has designer role."""
    agents = create_agents()
    designer = agents["designer"]

    assert designer.id == "designer"
    assert designer.name == "Designer"
    assert designer.role == "designer"


def test_designer_agent_loop_period() -> None:
    """Verify Designer has loop_period=60 (medium, design work)."""
    agents = create_agents()
    designer = agents["designer"]
    assert designer.loop_period == 60


def test_designer_agent_labels() -> None:
    """Verify Designer has Design label."""
    agents = create_agents()
    designer = agents["designer"]

    assert "SoftwareTeam" in designer.labels
    assert "Design" in designer.labels


def test_designer_agent_genome() -> None:
    """Verify Designer has correct initial genome values."""
    agents = create_agents()
    designer = agents["designer"]

    assert designer.genome["creativity"] == 0.8
    assert designer.genome["attention_to_detail"] == 0.7
    assert designer.genome["responsiveness"] == 0.7
    assert designer.genome["user_empathy"] == 0.8
    assert designer.genome["visual_design"] == 0.75


def test_designer_agent_has_policy() -> None:
    """Verify Designer has a working policy callable."""
    agents = create_agents()
    designer = agents["designer"]

    assert callable(designer.policy)
    result = designer.policy([], designer.genome, {})
    assert isinstance(result, list)


def test_create_world_has_agents() -> None:
    """Verify create_world populates world.agents."""
    world = create_world()
    assert len(world.agents) == 4
    assert "pm" in world.agents
    assert "dev1" in world.agents
    assert "dev2" in world.agents
    assert "designer" in world.agents


# ============================================================================
# POLICY BEHAVIOR TESTS
# ============================================================================


def test_pm_policy_assigns_feature_request() -> None:
    """Verify PM's policy creates task_assignment from feature_request."""
    feature_request = Particle(
        id="test_feature",
        particle_type="feature_request",
        payload={"feature_type": "dashboard", "priority": "high"},
        source_id="external",
        dest_id="pm",
        link_id="",
    )

    genome = {"prioritization": 0.8}
    internal_state: dict = {}

    result = pm_policy([feature_request], genome, internal_state)

    tasks = [p for p in result if p.particle_type == "task_assignment"]
    assert len(tasks) == 1
    assert tasks[0].source_id == "pm"
    assert tasks[0].dest_id in ("dev1", "dev2")


def test_pm_policy_assigns_bug_report() -> None:
    """Verify PM's policy creates task_assignment from bug_report."""
    bug_report = Particle(
        id="test_bug",
        particle_type="bug_report",
        payload={"severity": "critical", "component": "auth"},
        source_id="external",
        dest_id="pm",
        link_id="",
    )

    genome = {"prioritization": 0.8}
    internal_state: dict = {}

    result = pm_policy([bug_report], genome, internal_state)

    tasks = [p for p in result if p.particle_type == "task_assignment"]
    assert len(tasks) == 1
    assert tasks[0].payload["priority"] == "high"  # Critical bug -> high priority


def test_pm_policy_balances_workload() -> None:
    """Verify PM's policy balances task assignment between developers."""
    requests = [
        Particle(
            id=f"feature_{i}",
            particle_type="feature_request",
            payload={"feature_type": "dashboard", "priority": "medium"},
            source_id="external",
            dest_id="pm",
            link_id="",
        )
        for i in range(4)
    ]

    genome = {"prioritization": 0.8}
    internal_state: dict = {}

    result = pm_policy(requests, genome, internal_state)

    dev1_tasks = [p for p in result if p.dest_id == "dev1"]
    dev2_tasks = [p for p in result if p.dest_id == "dev2"]

    # Should be roughly balanced
    assert len(dev1_tasks) == 2
    assert len(dev2_tasks) == 2


def test_dev_policy_produces_code() -> None:
    """Verify developer's policy produces code_submission from task_assignment."""
    task = Particle(
        id="test_task",
        particle_type="task_assignment",
        payload={"request_type": "bug_report", "priority": "high"},
        source_id="pm",
        dest_id="dev1",
        link_id="pm_to_dev1",
    )

    genome = {"coding_speed": 0.7, "code_quality": 0.8, "collaboration": 0.3}
    internal_state = {"agent_id": "dev1"}

    result = dev_policy([task], genome, internal_state)

    code = [p for p in result if p.particle_type == "code_submission"]
    assert len(code) >= 1
    assert code[0].source_id == "dev1"
    assert code[0].dest_id == "dev2"  # Peer review


def test_dev_policy_requests_design() -> None:
    """Verify developer's policy requests design spec for feature requests."""
    task = Particle(
        id="test_task",
        particle_type="task_assignment",
        payload={"request_type": "feature_request", "priority": "medium"},
        source_id="pm",
        dest_id="dev1",
        link_id="pm_to_dev1",
    )

    # High collaboration trait should request design
    genome = {"coding_speed": 0.7, "code_quality": 0.8, "collaboration": 0.8}
    internal_state = {"agent_id": "dev1"}

    result = dev_policy([task], genome, internal_state)

    design_requests = [p for p in result if p.particle_type == "design_request"]
    assert len(design_requests) >= 1
    assert design_requests[0].dest_id == "designer"


def test_dev_policy_reviews_code() -> None:
    """Verify developer's policy provides code review for peer's code."""
    code = Particle(
        id="test_code",
        particle_type="code_submission",
        payload={"quality": 0.8, "author": "dev2"},
        source_id="dev2",
        dest_id="dev1",
        link_id="dev2_to_dev1",
    )

    genome = {"review_thoroughness": 0.8}
    internal_state = {"agent_id": "dev1"}

    result = dev_policy([code], genome, internal_state)

    reviews = [p for p in result if p.particle_type == "code_review"]
    assert len(reviews) == 1
    assert reviews[0].source_id == "dev1"
    assert reviews[0].dest_id == "dev2"


def test_dev_policy_notifies_completion() -> None:
    """Verify developer's policy notifies PM when code is approved."""
    review = Particle(
        id="test_review",
        particle_type="code_review",
        payload={"code_id": "code_1", "approved": True, "reviewer": "dev2"},
        source_id="dev2",
        dest_id="dev1",
        link_id="dev2_to_dev1",
    )

    genome = {}
    internal_state = {"agent_id": "dev1"}

    result = dev_policy([review], genome, internal_state)

    notices = [p for p in result if p.particle_type == "completion_notice"]
    assert len(notices) == 1
    assert notices[0].dest_id == "pm"


def test_designer_policy_creates_spec() -> None:
    """Verify designer's policy creates design_spec from design_request."""
    request = Particle(
        id="test_design_req",
        particle_type="design_request",
        payload={"task": {"feature_type": "dashboard"}, "requested_by": "dev1"},
        source_id="dev1",
        dest_id="designer",
        link_id="dev1_to_designer",
    )

    genome = {"creativity": 0.8, "attention_to_detail": 0.7}
    internal_state: dict = {}

    result = designer_policy([request], genome, internal_state)

    specs = [p for p in result if p.particle_type == "design_spec"]
    assert len(specs) == 1
    assert specs[0].source_id == "designer"
    assert specs[0].dest_id == "dev1"
    assert specs[0].payload["deliverable"] in DESIGN_DELIVERABLES


def test_designer_policy_responds_to_feedback() -> None:
    """Verify designer's policy responds to feedback requests."""
    feedback_req = Particle(
        id="test_feedback_req",
        particle_type="feedback_request",
        payload={"requested_by": "dev2"},
        source_id="dev2",
        dest_id="designer",
        link_id="dev2_to_designer",
    )

    genome = {"responsiveness": 0.8}
    internal_state: dict = {}

    result = designer_policy([feedback_req], genome, internal_state)

    responses = [p for p in result if p.particle_type == "feedback_response"]
    assert len(responses) == 1
    assert responses[0].dest_id == "dev2"


# ============================================================================
# WORLD CREATION TESTS
# ============================================================================


def test_create_software_team_world_complete() -> None:
    """Verify create_software_team_world creates a complete world."""
    world = create_software_team_world()

    assert len(world.agents) == 4
    assert len(world.links) == 10
    assert len(world.labels) == 5
    assert len(world.external_inputs) == 2


def test_create_world_alias() -> None:
    """Verify create_world is an alias for create_software_team_world."""
    world1 = create_software_team_world()
    world2 = create_world()

    assert len(world1.agents) == len(world2.agents)
    assert len(world1.links) == len(world2.links)


def test_world_loads_without_framework_changes() -> None:
    """Verify world can be created using only standard loopengine imports."""
    # This test verifies that no framework code modifications are needed
    # by creating the world and running basic assertions
    world = create_world()

    # All agents have valid policies
    for agent in world.agents.values():
        assert callable(agent.policy)
        # Policy can be called without errors
        result = agent.policy([], agent.genome, dict(agent.internal_state))
        assert isinstance(result, list)

    # All links reference valid agents
    for link in world.links.values():
        assert link.source_id in world.agents
        assert link.dest_id in world.agents

    # All external inputs target valid agents
    for ext_input in world.external_inputs:
        assert ext_input.target_agent_id in world.agents

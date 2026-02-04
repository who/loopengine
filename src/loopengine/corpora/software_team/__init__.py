"""The Software Team corpus: A small software team with four members.

This is the second implementation corpus for LoopEngine, demonstrating
agents, links, labels, and external inputs in a software development context.
"""

from __future__ import annotations

import random
from typing import Any

from loopengine.model import (
    Agent,
    ExternalInput,
    Label,
    LabelContext,
    Link,
    LinkType,
    Particle,
    World,
)

# Feature types and bug severities for payload generation
FEATURE_TYPES = ["dashboard", "auth", "notifications", "reports", "integrations", "settings"]
BUG_SEVERITIES = ["critical", "major", "minor", "trivial"]
DESIGN_DELIVERABLES = ["mockup", "prototype", "spec", "user_flow", "component_library"]
CODE_LANGUAGES = ["python", "typescript", "sql", "yaml", "dockerfile"]


def create_links() -> dict[str, Link]:
    """Create the 10 links between software team agents.

    Links:
    - pm_to_dev1: HIERARCHICAL (PM -> Dev1)
    - pm_to_dev2: HIERARCHICAL (PM -> Dev2)
    - dev1_to_pm: HIERARCHICAL upward (Dev1 -> PM)
    - dev2_to_pm: HIERARCHICAL upward (Dev2 -> PM)
    - dev1_to_dev2: PEER (Dev1 <-> Dev2)
    - dev2_to_dev1: PEER (Dev2 <-> Dev1)
    - designer_to_dev1: SERVICE (Designer -> Dev1)
    - designer_to_dev2: SERVICE (Designer -> Dev2)
    - dev1_to_designer: SERVICE (Dev1 -> Designer)
    - dev2_to_designer: SERVICE (Dev2 -> Designer)

    Returns:
        dict[str, Link]: Links keyed by id.
    """
    links = {}

    # PM -> Dev1 (HIERARCHICAL downward)
    links["pm_to_dev1"] = Link(
        id="pm_to_dev1",
        source_id="pm",
        dest_id="dev1",
        link_type=LinkType.HIERARCHICAL,
        properties={
            "authority_scope": ["task_assignment", "priority_setting", "deadline_management"],
            "autonomy_granted": 0.6,
            "fitness_definition": ["delivery_speed", "code_quality", "collaboration"],
            "flow_types": ["task_assignment", "priority_update"],
        },
    )

    # PM -> Dev2 (HIERARCHICAL downward)
    links["pm_to_dev2"] = Link(
        id="pm_to_dev2",
        source_id="pm",
        dest_id="dev2",
        link_type=LinkType.HIERARCHICAL,
        properties={
            "authority_scope": ["task_assignment", "priority_setting", "deadline_management"],
            "autonomy_granted": 0.6,
            "fitness_definition": ["delivery_speed", "code_quality", "collaboration"],
            "flow_types": ["task_assignment", "priority_update"],
        },
    )

    # Dev1 -> PM (HIERARCHICAL upward)
    links["dev1_to_pm"] = Link(
        id="dev1_to_pm",
        source_id="dev1",
        dest_id="pm",
        link_type=LinkType.HIERARCHICAL,
        properties={
            "flow_types": ["status_update", "blocker_report", "completion_notice"],
        },
    )

    # Dev2 -> PM (HIERARCHICAL upward)
    links["dev2_to_pm"] = Link(
        id="dev2_to_pm",
        source_id="dev2",
        dest_id="pm",
        link_type=LinkType.HIERARCHICAL,
        properties={
            "flow_types": ["status_update", "blocker_report", "completion_notice"],
        },
    )

    # Dev1 <-> Dev2 (PEER: code review, pair programming)
    links["dev1_to_dev2"] = Link(
        id="dev1_to_dev2",
        source_id="dev1",
        dest_id="dev2",
        link_type=LinkType.PEER,
        properties={
            "flow_types": ["code_review_request", "pair_session", "knowledge_share"],
            "bandwidth": 1.0,
        },
    )

    links["dev2_to_dev1"] = Link(
        id="dev2_to_dev1",
        source_id="dev2",
        dest_id="dev1",
        link_type=LinkType.PEER,
        properties={
            "flow_types": ["code_review_request", "pair_session", "knowledge_share"],
            "bandwidth": 1.0,
        },
    )

    # Designer -> Dev1 (SERVICE: design deliverables)
    links["designer_to_dev1"] = Link(
        id="designer_to_dev1",
        source_id="designer",
        dest_id="dev1",
        link_type=LinkType.SERVICE,
        properties={
            "flow_types": ["design_spec", "mockup", "feedback_response"],
            "bandwidth": 1.0,
        },
    )

    # Designer -> Dev2 (SERVICE: design deliverables)
    links["designer_to_dev2"] = Link(
        id="designer_to_dev2",
        source_id="designer",
        dest_id="dev2",
        link_type=LinkType.SERVICE,
        properties={
            "flow_types": ["design_spec", "mockup", "feedback_response"],
            "bandwidth": 1.0,
        },
    )

    # Dev1 -> Designer (SERVICE: feedback requests)
    links["dev1_to_designer"] = Link(
        id="dev1_to_designer",
        source_id="dev1",
        dest_id="designer",
        link_type=LinkType.SERVICE,
        properties={
            "flow_types": ["design_request", "feedback_request", "clarification"],
            "bandwidth": 1.0,
        },
    )

    # Dev2 -> Designer (SERVICE: feedback requests)
    links["dev2_to_designer"] = Link(
        id="dev2_to_designer",
        source_id="dev2",
        dest_id="designer",
        link_type=LinkType.SERVICE,
        properties={
            "flow_types": ["design_request", "feedback_request", "clarification"],
            "bandwidth": 1.0,
        },
    )

    return links


def generate_feature_request_payload() -> dict[str, Any]:
    """Generate a random feature request payload.

    Returns:
        dict: Feature details with type, description, and priority.
    """
    return {
        "feature_type": random.choice(FEATURE_TYPES),
        "description": f"Implement {random.choice(FEATURE_TYPES)} feature",
        "priority": random.choice(["high", "medium", "low"]),
        "estimated_effort": random.choice(["small", "medium", "large"]),
    }


def generate_bug_report_payload() -> dict[str, Any]:
    """Generate a random bug report payload.

    Returns:
        dict: Bug details with severity, component, and reproduction steps.
    """
    return {
        "severity": random.choice(BUG_SEVERITIES),
        "component": random.choice(FEATURE_TYPES),
        "title": f"Bug in {random.choice(FEATURE_TYPES)}",
        "reproduced": random.choice([True, False]),
    }


def sprint_schedule(tick: int) -> float:
    """Return the rate multiplier for incoming work based on tick.

    Sprint start (first 100 ticks) has higher feature request rate.
    Sprint end (ticks 900-1000) has higher bug report rate.

    Args:
        tick: Current simulation tick.

    Returns:
        float: Rate multiplier (1.5 during sprint start, 1.0 otherwise).
    """
    if tick <= 100:
        return 1.5  # Sprint planning spike
    if tick >= 900:
        return 0.5  # Winding down for release
    return 1.0


def create_external_inputs() -> list[ExternalInput]:
    """Create external inputs for the software team.

    External inputs:
    - feature_requests: New feature requests targeting PM at rate 0.03/tick
    - bug_reports: Bug reports targeting PM at rate 0.02/tick

    Returns:
        list[ExternalInput]: External inputs for the world.
    """
    return [
        ExternalInput(
            name="feature_requests",
            target_agent_id="pm",
            rate=0.03,
            variance=0.2,
            particle_type="feature_request",
            payload_generator=generate_feature_request_payload,
            schedule=sprint_schedule,
        ),
        ExternalInput(
            name="bug_reports",
            target_agent_id="pm",
            rate=0.02,
            variance=0.3,
            particle_type="bug_report",
            payload_generator=generate_bug_report_payload,
            schedule=lambda tick: 1.0,  # Constant rate
        ),
    ]


def create_labels() -> dict[str, Label]:
    """Create the 5 labels for the software team.

    Labels:
    - SoftwareTeam: Global team context with development norms
    - Engineering: Technical context for developers
    - Design: Design and UX context
    - Management: Administrative context for PM
    - Codebase: Shared code repository context

    Returns:
        dict[str, Label]: Labels keyed by name.
    """
    labels = {}

    # SoftwareTeam: global context for the entire team
    labels["SoftwareTeam"] = Label(
        name="SoftwareTeam",
        context=LabelContext(
            constraints=["code_review_required", "documentation_standards", "test_coverage_80"],
            resources=["git_repository", "ci_pipeline", "issue_tracker", "chat_system"],
            norms=["daily_standup", "sprint_planning", "retrospective"],
        ),
    )

    # Engineering: developer context
    labels["Engineering"] = Label(
        name="Engineering",
        context=LabelContext(
            constraints=["code_style_guide", "security_review", "performance_budget"],
            resources=["ide", "debugger", "test_framework", "local_dev_env"],
            norms=["pair_programming", "code_review", "knowledge_sharing"],
        ),
    )

    # Design: designer context
    labels["Design"] = Label(
        name="Design",
        context=LabelContext(
            constraints=["accessibility_standards", "brand_guidelines", "responsive_design"],
            resources=["figma", "design_system", "user_research_data"],
            norms=["design_review", "user_testing", "iteration"],
        ),
    )

    # Management: PM context
    labels["Management"] = Label(
        name="Management",
        context=LabelContext(
            resources=["roadmap", "stakeholder_access", "analytics", "backlog"],
            norms=["prioritization", "stakeholder_communication", "team_health_check"],
        ),
    )

    # Codebase: shared repository context
    labels["Codebase"] = Label(
        name="Codebase",
        context=LabelContext(
            constraints=["branch_naming", "commit_message_format", "merge_approval"],
            resources=["main_branch", "feature_branches", "release_tags"],
        ),
    )

    return labels


def pm_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """PM's policy: receive requests, prioritize, and assign work to developers.

    On SENSE: observe incoming feature requests and bug reports
    On DECIDE: prioritize based on urgency and team capacity
    On ACT: assign tasks to dev1 or dev2, balance workload

    Args:
        sensed_inputs: Particles received (feature_requests, bug_reports, status_updates)
        genome: PM's genome traits
        internal_state: Current internal state

    Returns:
        list[Particle]: Output particles (task_assignments, priority_updates)
    """
    outputs: list[Particle] = []

    # Initialize tracking metrics
    if "tasks_assigned" not in internal_state:
        internal_state["tasks_assigned"] = 0
    if "dev1_workload" not in internal_state:
        internal_state["dev1_workload"] = 0
    if "dev2_workload" not in internal_state:
        internal_state["dev2_workload"] = 0
    if "blocked_items" not in internal_state:
        internal_state["blocked_items"] = 0

    for particle in sensed_inputs:
        if particle.particle_type in ("feature_request", "bug_report"):
            # Decide which developer to assign based on workload balancing
            prioritization = genome.get("prioritization", 0.7)

            # Balance workload between developers
            if internal_state["dev1_workload"] <= internal_state["dev2_workload"]:
                target_dev = "dev1"
                target_link = "pm_to_dev1"
                internal_state["dev1_workload"] += 1
            else:
                target_dev = "dev2"
                target_link = "pm_to_dev2"
                internal_state["dev2_workload"] += 1

            # Assign priority based on prioritization trait
            if particle.particle_type == "bug_report":
                severity = particle.payload.get("severity", "minor")
                if severity == "critical":
                    priority = "high"
                elif severity == "major":
                    priority = "high" if prioritization > 0.6 else "medium"
                else:
                    priority = "medium" if prioritization > 0.5 else "low"
            else:
                incoming_priority = particle.payload.get("priority", "medium")
                priority = incoming_priority

            outputs.append(
                Particle(
                    id=f"task_{particle.id}",
                    particle_type="task_assignment",
                    payload={
                        "original_request": particle.payload,
                        "request_type": particle.particle_type,
                        "priority": priority,
                        "assigned_by": "pm",
                    },
                    source_id="pm",
                    dest_id=target_dev,
                    link_id=target_link,
                )
            )
            internal_state["tasks_assigned"] += 1

        elif particle.particle_type == "completion_notice":
            # Track completed work
            completed_by = particle.payload.get("completed_by", "")
            if completed_by == "dev1":
                internal_state["dev1_workload"] = max(0, internal_state["dev1_workload"] - 1)
            elif completed_by == "dev2":
                internal_state["dev2_workload"] = max(0, internal_state["dev2_workload"] - 1)

        elif particle.particle_type == "blocker_report":
            internal_state["blocked_items"] += 1

    return outputs


def dev_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """Developer policy: receive tasks, write code, request reviews.

    On SENSE: observe task assignments and design specs
    On DECIDE: plan implementation approach
    On ACT: produce code, request code reviews, report status

    Args:
        sensed_inputs: Particles received (task_assignment, design_spec, code_review)
        genome: Developer's genome traits
        internal_state: Current internal state

    Returns:
        list[Particle]: Output particles (code, review_requests, status_updates)
    """
    outputs: list[Particle] = []

    # Initialize tracking metrics
    if "tasks_completed" not in internal_state:
        internal_state["tasks_completed"] = 0
    if "code_quality_scores" not in internal_state:
        internal_state["code_quality_scores"] = []
    if "reviews_requested" not in internal_state:
        internal_state["reviews_requested"] = 0
    if "agent_id" not in internal_state:
        internal_state["agent_id"] = "dev1"  # Default, should be set by simulation

    agent_id = internal_state.get("agent_id", "dev1")
    peer_id = "dev2" if agent_id == "dev1" else "dev1"
    peer_link = f"{agent_id}_to_{peer_id}"
    pm_link = f"{agent_id}_to_pm"
    designer_link = f"{agent_id}_to_designer"

    for particle in sensed_inputs:
        if particle.particle_type == "task_assignment":
            # Work on the task based on coding_speed and code_quality traits
            coding_speed = genome.get("coding_speed", 0.7)
            code_quality = genome.get("code_quality", 0.8)

            # Quality is influenced by speed tradeoff
            actual_quality = code_quality * (1 - (coding_speed - 0.5) * 0.2)
            actual_quality = max(0.5, min(1.0, actual_quality))

            internal_state["code_quality_scores"].append(actual_quality)

            # Check if design spec is needed
            request_type = particle.payload.get("request_type", "feature_request")
            collaboration = genome.get("collaboration", 0.6)

            if request_type == "feature_request" and collaboration > 0.5:
                # Request design spec first
                outputs.append(
                    Particle(
                        id=f"design_req_{particle.id}",
                        particle_type="design_request",
                        payload={
                            "task": particle.payload,
                            "requested_by": agent_id,
                        },
                        source_id=agent_id,
                        dest_id="designer",
                        link_id=designer_link,
                    )
                )

            # Produce code
            outputs.append(
                Particle(
                    id=f"code_{particle.id}",
                    particle_type="code_submission",
                    payload={
                        "task": particle.payload,
                        "quality": actual_quality,
                        "language": random.choice(CODE_LANGUAGES),
                        "author": agent_id,
                    },
                    source_id=agent_id,
                    dest_id=peer_id,
                    link_id=peer_link,
                )
            )
            internal_state["reviews_requested"] += 1

        elif particle.particle_type == "code_review_request":
            # Review code from peer
            review_quality = genome.get("review_thoroughness", 0.7)

            # Provide feedback
            outputs.append(
                Particle(
                    id=f"review_{particle.id}",
                    particle_type="code_review",
                    payload={
                        "code_id": particle.id,
                        "approved": review_quality > 0.5,
                        "comments": "LGTM" if review_quality > 0.6 else "Needs changes",
                        "reviewer": agent_id,
                    },
                    source_id=agent_id,
                    dest_id=particle.source_id,
                    link_id=peer_link,
                )
            )

        elif particle.particle_type == "code_submission":
            # Received code for review - treat as review request
            review_quality = genome.get("review_thoroughness", 0.7)

            outputs.append(
                Particle(
                    id=f"review_{particle.id}",
                    particle_type="code_review",
                    payload={
                        "code_id": particle.id,
                        "approved": review_quality > 0.5,
                        "comments": "LGTM" if review_quality > 0.6 else "Needs changes",
                        "reviewer": agent_id,
                    },
                    source_id=agent_id,
                    dest_id=particle.source_id,
                    link_id=peer_link,
                )
            )

        elif particle.particle_type == "code_review":
            # Received review feedback
            if particle.payload.get("approved", False):
                internal_state["tasks_completed"] += 1

                # Notify PM of completion
                outputs.append(
                    Particle(
                        id=f"complete_{particle.id}",
                        particle_type="completion_notice",
                        payload={
                            "completed_by": agent_id,
                            "code_id": particle.payload.get("code_id", ""),
                        },
                        source_id=agent_id,
                        dest_id="pm",
                        link_id=pm_link,
                    )
                )

        elif particle.particle_type == "design_spec":
            # Received design specification - can proceed with implementation
            internal_state["design_specs_received"] = (
                internal_state.get("design_specs_received", 0) + 1
            )

    return outputs


def designer_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """Designer policy: receive requests, create designs, respond to feedback.

    On SENSE: observe design requests and feedback
    On DECIDE: prioritize design work
    On ACT: produce design specs, mockups, respond to clarifications

    Args:
        sensed_inputs: Particles received (design_request, feedback_request)
        genome: Designer's genome traits
        internal_state: Current internal state

    Returns:
        list[Particle]: Output particles (design_spec, mockup, feedback_response)
    """
    outputs: list[Particle] = []

    # Initialize tracking metrics
    if "designs_created" not in internal_state:
        internal_state["designs_created"] = 0
    if "feedback_responses" not in internal_state:
        internal_state["feedback_responses"] = 0

    for particle in sensed_inputs:
        if particle.particle_type == "design_request":
            # Create design based on creativity and attention_to_detail traits
            creativity = genome.get("creativity", 0.7)
            attention_to_detail = genome.get("attention_to_detail", 0.8)

            design_quality = (creativity + attention_to_detail) / 2
            requested_by = particle.payload.get("requested_by", "dev1")
            target_link = f"designer_to_{requested_by}"

            outputs.append(
                Particle(
                    id=f"design_{particle.id}",
                    particle_type="design_spec",
                    payload={
                        "task": particle.payload.get("task", {}),
                        "deliverable": random.choice(DESIGN_DELIVERABLES),
                        "quality": design_quality,
                        "designer": "designer",
                    },
                    source_id="designer",
                    dest_id=requested_by,
                    link_id=target_link,
                )
            )
            internal_state["designs_created"] += 1

        elif particle.particle_type == "feedback_request":
            # Respond to feedback requests
            responsiveness = genome.get("responsiveness", 0.7)
            requested_by = particle.payload.get("requested_by", "dev1")
            target_link = f"designer_to_{requested_by}"

            if responsiveness > 0.5:
                outputs.append(
                    Particle(
                        id=f"feedback_{particle.id}",
                        particle_type="feedback_response",
                        payload={
                            "original_request": particle.payload,
                            "response": "Design clarification provided",
                        },
                        source_id="designer",
                        dest_id=requested_by,
                        link_id=target_link,
                    )
                )
                internal_state["feedback_responses"] += 1

        elif particle.particle_type == "clarification":
            # Handle clarification requests
            requested_by = particle.source_id
            target_link = f"designer_to_{requested_by}"

            outputs.append(
                Particle(
                    id=f"clarify_{particle.id}",
                    particle_type="feedback_response",
                    payload={
                        "original_request": particle.payload,
                        "response": "Clarification provided",
                    },
                    source_id="designer",
                    dest_id=requested_by,
                    link_id=target_link,
                )
            )
            internal_state["feedback_responses"] += 1

    return outputs


def create_agents() -> dict[str, Agent]:
    """Create the 4 agents for the software team.

    Agents:
    - PM: loop_period=200, Management label, task prioritization
    - Dev1: loop_period=40, Engineering label, coding
    - Dev2: loop_period=40, Engineering label, coding
    - Designer: loop_period=60, Design label, design work

    Returns:
        dict[str, Agent]: Agents keyed by id.
    """
    agents = {}

    # PM - Product Manager
    agents["pm"] = Agent(
        id="pm",
        name="PM",
        role="product_manager",
        genome={
            "prioritization": 0.8,
            "stakeholder_management": 0.7,
            "team_coordination": 0.8,
            "decisiveness": 0.6,
            "communication": 0.8,
        },
        labels={"SoftwareTeam", "Management"},
        loop_period=200,
        policy=pm_policy,
    )

    # Dev1 - Developer 1
    agents["dev1"] = Agent(
        id="dev1",
        name="Dev1",
        role="developer",
        genome={
            "coding_speed": 0.7,
            "code_quality": 0.8,
            "collaboration": 0.7,
            "review_thoroughness": 0.7,
            "debugging_skill": 0.6,
        },
        labels={"SoftwareTeam", "Engineering", "Codebase"},
        loop_period=40,
        policy=dev_policy,
        internal_state={"agent_id": "dev1"},
    )

    # Dev2 - Developer 2
    agents["dev2"] = Agent(
        id="dev2",
        name="Dev2",
        role="developer",
        genome={
            "coding_speed": 0.6,
            "code_quality": 0.85,
            "collaboration": 0.8,
            "review_thoroughness": 0.8,
            "debugging_skill": 0.7,
        },
        labels={"SoftwareTeam", "Engineering", "Codebase"},
        loop_period=40,
        policy=dev_policy,
        internal_state={"agent_id": "dev2"},
    )

    # Designer
    agents["designer"] = Agent(
        id="designer",
        name="Designer",
        role="designer",
        genome={
            "creativity": 0.8,
            "attention_to_detail": 0.7,
            "responsiveness": 0.7,
            "user_empathy": 0.8,
            "visual_design": 0.75,
        },
        labels={"SoftwareTeam", "Design"},
        loop_period=60,
        policy=designer_policy,
    )

    return agents


def create_software_team_world() -> World:
    """Create the software team world with agents, links, labels, and external inputs.

    Returns:
        World: A world containing the complete software team simulation.
    """
    world = World()
    world.agents = create_agents()
    world.links = create_links()
    world.labels = create_labels()
    world.external_inputs = create_external_inputs()
    return world


# Alias for consistency with other corpora
create_world = create_software_team_world

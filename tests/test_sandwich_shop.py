"""Tests for the Sandwich Shop corpus."""

from loopengine.corpora.sandwich_shop import (
    EXTRAS,
    SANDWICH_TYPES,
    SPECIAL_REQUESTS,
    create_external_inputs,
    create_labels,
    create_links,
    create_world,
    customer_arrival_schedule,
    generate_customer_order_payload,
)
from loopengine.model import LinkType


def test_links_count() -> None:
    """Verify 6 links are created."""
    links = create_links()
    assert len(links) == 6


def test_links_ids() -> None:
    """Verify all link IDs match PRD naming."""
    links = create_links()
    expected_ids = {
        "maria_to_tom",
        "tom_to_maria",
        "maria_to_alex",
        "alex_to_maria",
        "alex_to_tom",
        "tom_to_alex",
    }
    assert set(links.keys()) == expected_ids


def test_links_are_directional() -> None:
    """Verify links are directional (alex_to_tom separate from tom_to_alex)."""
    links = create_links()

    alex_to_tom = links["alex_to_tom"]
    tom_to_alex = links["tom_to_alex"]

    # Different source/dest
    assert alex_to_tom.source_id == "alex"
    assert alex_to_tom.dest_id == "tom"
    assert tom_to_alex.source_id == "tom"
    assert tom_to_alex.dest_id == "alex"

    # Different flow types
    assert alex_to_tom.properties["flow_types"] == ["order_ticket"]
    assert tom_to_alex.properties["flow_types"] == ["finished_sandwich"]


def test_maria_to_tom_hierarchical() -> None:
    """Verify maria_to_tom has HIERARCHICAL type with authority_scope and autonomy_granted."""
    links = create_links()
    link = links["maria_to_tom"]

    assert link.link_type == LinkType.HIERARCHICAL
    assert link.source_id == "maria"
    assert link.dest_id == "tom"
    assert "authority_scope" in link.properties
    assert link.properties["authority_scope"] == ["recipe_standards", "supply_priorities"]
    assert "autonomy_granted" in link.properties
    assert link.properties["autonomy_granted"] == 0.5
    assert "fitness_definition" in link.properties
    assert "flow_types" in link.properties


def test_tom_to_maria_hierarchical_upward() -> None:
    """Verify tom_to_maria is HIERARCHICAL upward with flow_types."""
    links = create_links()
    link = links["tom_to_maria"]

    assert link.link_type == LinkType.HIERARCHICAL
    assert link.source_id == "tom"
    assert link.dest_id == "maria"
    assert "flow_types" in link.properties
    assert "status_report" in link.properties["flow_types"]
    assert "stockout_alert" in link.properties["flow_types"]


def test_maria_to_alex_hierarchical() -> None:
    """Verify maria_to_alex has HIERARCHICAL type with authority_scope and autonomy_granted."""
    links = create_links()
    link = links["maria_to_alex"]

    assert link.link_type == LinkType.HIERARCHICAL
    assert link.source_id == "maria"
    assert link.dest_id == "alex"
    assert "authority_scope" in link.properties
    assert link.properties["authority_scope"] == ["service_standards", "upselling_policy"]
    assert "autonomy_granted" in link.properties
    assert link.properties["autonomy_granted"] == 0.4


def test_alex_to_maria_hierarchical_upward() -> None:
    """Verify alex_to_maria is HIERARCHICAL upward with flow_types."""
    links = create_links()
    link = links["alex_to_maria"]

    assert link.link_type == LinkType.HIERARCHICAL
    assert link.source_id == "alex"
    assert link.dest_id == "maria"
    assert "flow_types" in link.properties
    assert "revenue_report" in link.properties["flow_types"]
    assert "status_report" in link.properties["flow_types"]


def test_alex_to_tom_service() -> None:
    """Verify alex_to_tom has SERVICE type with flow_types and bandwidth."""
    links = create_links()
    link = links["alex_to_tom"]

    assert link.link_type == LinkType.SERVICE
    assert link.source_id == "alex"
    assert link.dest_id == "tom"
    assert "flow_types" in link.properties
    assert link.properties["flow_types"] == ["order_ticket"]
    assert "bandwidth" in link.properties
    assert link.properties["bandwidth"] == 1.0


def test_tom_to_alex_service() -> None:
    """Verify tom_to_alex has SERVICE type with flow_types and bandwidth."""
    links = create_links()
    link = links["tom_to_alex"]

    assert link.link_type == LinkType.SERVICE
    assert link.source_id == "tom"
    assert link.dest_id == "alex"
    assert "flow_types" in link.properties
    assert link.properties["flow_types"] == ["finished_sandwich"]
    assert "bandwidth" in link.properties
    assert link.properties["bandwidth"] == 1.0


def test_create_world_has_links() -> None:
    """Verify create_world populates world.links."""
    world = create_world()
    assert len(world.links) == 6
    assert "maria_to_tom" in world.links


# ============================================================================
# LABEL TESTS (PRD Section 9.4)
# ============================================================================


def test_labels_count() -> None:
    """Verify 5 labels are created."""
    labels = create_labels()
    assert len(labels) == 5


def test_labels_names() -> None:
    """Verify all label names match PRD."""
    labels = create_labels()
    expected_names = {"SandwichShop", "FrontLine", "Kitchen", "Management", "Register"}
    assert set(labels.keys()) == expected_names


def test_sandwichshop_label_constraints() -> None:
    """Verify SandwichShop label has health_code in constraints per PRD."""
    labels = create_labels()
    label = labels["SandwichShop"]

    assert label.name == "SandwichShop"
    assert "health_code" in label.context.constraints
    assert "operating_hours_10_to_8" in label.context.constraints
    assert "max_capacity_30" in label.context.constraints


def test_sandwichshop_label_resources() -> None:
    """Verify SandwichShop label has correct resources per PRD."""
    labels = create_labels()
    label = labels["SandwichShop"]

    assert "ingredient_supply" in label.context.resources
    assert "POS_system" in label.context.resources
    assert "kitchen_equipment" in label.context.resources


def test_sandwichshop_label_norms() -> None:
    """Verify SandwichShop label has correct norms per PRD."""
    labels = create_labels()
    label = labels["SandwichShop"]

    assert "FIFO_orders" in label.context.norms
    assert "customer_greeting" in label.context.norms
    assert "clean_as_you_go" in label.context.norms


def test_frontline_label() -> None:
    """Verify FrontLine label has customer-facing constraints and norms."""
    labels = create_labels()
    label = labels["FrontLine"]

    assert label.name == "FrontLine"
    assert "customer_facing_appearance" in label.context.constraints
    assert "friendly_demeanor" in label.context.norms
    assert "no_phone_use" in label.context.norms


def test_kitchen_label() -> None:
    """Verify Kitchen label has food safety constraints and equipment resources."""
    labels = create_labels()
    label = labels["Kitchen"]

    assert label.name == "Kitchen"
    assert "food_safety_gloves" in label.context.constraints
    assert "temperature_monitoring" in label.context.constraints
    assert "grill" in label.context.resources
    assert "prep_station" in label.context.resources
    assert "cold_storage" in label.context.resources


def test_management_label() -> None:
    """Verify Management label has administrative resources and norms."""
    labels = create_labels()
    label = labels["Management"]

    assert label.name == "Management"
    assert "supplier_contacts" in label.context.resources
    assert "POS_admin" in label.context.resources
    assert "scheduling_system" in label.context.resources
    assert "daily_inventory_check" in label.context.norms
    assert "weekly_supply_order" in label.context.norms


def test_register_label() -> None:
    """Verify Register label has POS resources and cash handling constraint."""
    labels = create_labels()
    label = labels["Register"]

    assert label.name == "Register"
    assert "POS_terminal" in label.context.resources
    assert "cash_drawer" in label.context.resources
    assert "cash_handling_policy" in label.context.constraints


def test_create_world_has_labels() -> None:
    """Verify create_world populates world.labels."""
    world = create_world()
    assert len(world.labels) == 5
    assert "SandwichShop" in world.labels
    assert "Kitchen" in world.labels


# ============================================================================
# EXTERNAL INPUT TESTS (PRD Section 9.5)
# ============================================================================


def test_external_inputs_count() -> None:
    """Verify 1 external input is created (customer_arrivals)."""
    inputs = create_external_inputs()
    assert len(inputs) == 1


def test_customer_arrivals_external_input() -> None:
    """Verify customer_arrivals external input targeting Alex with rate=0.05."""
    inputs = create_external_inputs()
    customer_arrivals = inputs[0]

    assert customer_arrivals.name == "customer_arrivals"
    assert customer_arrivals.target_agent_id == "alex"
    assert customer_arrivals.rate == 0.05
    assert customer_arrivals.variance == 0.3
    assert customer_arrivals.particle_type == "customer_order"


def test_customer_arrival_schedule_normal() -> None:
    """Verify schedule returns 1.0 outside lunch rush (ticks < 200 and > 400)."""
    assert customer_arrival_schedule(0) == 1.0
    assert customer_arrival_schedule(100) == 1.0
    assert customer_arrival_schedule(199) == 1.0
    assert customer_arrival_schedule(401) == 1.0
    assert customer_arrival_schedule(500) == 1.0


def test_customer_arrival_schedule_rush() -> None:
    """Verify schedule returns 2.0 during lunch rush (ticks 200-400)."""
    assert customer_arrival_schedule(200) == 2.0
    assert customer_arrival_schedule(250) == 2.0
    assert customer_arrival_schedule(300) == 2.0
    assert customer_arrival_schedule(400) == 2.0


def test_payload_generator_structure() -> None:
    """Verify payload generator returns dict with sandwich_type, extras, special_requests."""
    payload = generate_customer_order_payload()

    assert "sandwich_type" in payload
    assert "extras" in payload
    assert "special_requests" in payload


def test_payload_generator_sandwich_type() -> None:
    """Verify sandwich_type is from valid list."""
    for _ in range(20):  # Test multiple times due to randomness
        payload = generate_customer_order_payload()
        assert payload["sandwich_type"] in SANDWICH_TYPES


def test_payload_generator_extras() -> None:
    """Verify extras are from valid list and count is 0-2."""
    for _ in range(20):
        payload = generate_customer_order_payload()
        extras = payload["extras"]
        assert len(extras) <= 2
        for extra in extras:
            assert extra in EXTRAS


def test_payload_generator_special_requests() -> None:
    """Verify special_requests are from valid list and count is 0-1."""
    for _ in range(20):
        payload = generate_customer_order_payload()
        requests = payload["special_requests"]
        assert len(requests) <= 1
        for req in requests:
            assert req in SPECIAL_REQUESTS


def test_external_input_has_callable_schedule() -> None:
    """Verify external input schedule is callable and returns rate multiplier."""
    inputs = create_external_inputs()
    customer_arrivals = inputs[0]

    # Schedule should be callable
    assert callable(customer_arrivals.schedule)
    # Check it returns expected values
    assert customer_arrivals.schedule(100) == 1.0
    assert customer_arrivals.schedule(250) == 2.0


def test_external_input_has_callable_payload_generator() -> None:
    """Verify external input payload_generator is callable and returns dict."""
    inputs = create_external_inputs()
    customer_arrivals = inputs[0]

    # Payload generator should be callable
    assert callable(customer_arrivals.payload_generator)
    # Check it returns a dict
    payload = customer_arrivals.payload_generator()
    assert isinstance(payload, dict)


def test_create_world_has_external_inputs() -> None:
    """Verify create_world populates world.external_inputs."""
    world = create_world()
    assert len(world.external_inputs) == 1
    assert world.external_inputs[0].name == "customer_arrivals"

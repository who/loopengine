"""The Sandwich Shop corpus: A small sandwich shop with three employees.

This is the first implementation corpus for LoopEngine, demonstrating
agents, links, labels, and external inputs.
"""

from __future__ import annotations

import random
from typing import Any

from loopengine.model import ExternalInput, Label, LabelContext, Link, LinkType, World

# Sandwich types and extras for payload generation
SANDWICH_TYPES = ["BLT", "Club", "Reuben", "Turkey", "Veggie", "Philly"]
EXTRAS = ["extra_cheese", "avocado", "bacon", "pickles", "hot_peppers"]
SPECIAL_REQUESTS = ["no_mayo", "gluten_free_bread", "toasted", "cut_in_half", "extra_sauce"]


def create_links() -> dict[str, Link]:
    """Create the 6 links between sandwich shop agents per PRD section 9.3.

    Links:
    - maria_to_tom: HIERARCHICAL (Maria → Tom)
    - tom_to_maria: HIERARCHICAL upward (Tom → Maria)
    - maria_to_alex: HIERARCHICAL (Maria → Alex)
    - alex_to_maria: HIERARCHICAL upward (Alex → Maria)
    - alex_to_tom: SERVICE (Alex → Tom)
    - tom_to_alex: SERVICE (Tom → Alex)

    Returns:
        dict[str, Link]: Links keyed by id.
    """
    links = {}

    # Maria → Tom (HIERARCHICAL downward)
    links["maria_to_tom"] = Link(
        id="maria_to_tom",
        source_id="maria",
        dest_id="tom",
        link_type=LinkType.HIERARCHICAL,
        properties={
            "authority_scope": ["recipe_standards", "supply_priorities"],
            "autonomy_granted": 0.5,
            "fitness_definition": ["speed", "consistency", "waste"],
            "flow_types": ["directive"],
        },
    )

    # Tom → Maria (HIERARCHICAL upward)
    links["tom_to_maria"] = Link(
        id="tom_to_maria",
        source_id="tom",
        dest_id="maria",
        link_type=LinkType.HIERARCHICAL,
        properties={
            "flow_types": ["status_report", "stockout_alert"],
        },
    )

    # Maria → Alex (HIERARCHICAL downward)
    links["maria_to_alex"] = Link(
        id="maria_to_alex",
        source_id="maria",
        dest_id="alex",
        link_type=LinkType.HIERARCHICAL,
        properties={
            "authority_scope": ["service_standards", "upselling_policy"],
            "autonomy_granted": 0.4,
            "fitness_definition": ["throughput", "accuracy", "friendliness"],
            "flow_types": ["directive"],
        },
    )

    # Alex → Maria (HIERARCHICAL upward)
    links["alex_to_maria"] = Link(
        id="alex_to_maria",
        source_id="alex",
        dest_id="maria",
        link_type=LinkType.HIERARCHICAL,
        properties={
            "flow_types": ["revenue_report", "status_report"],
        },
    )

    # Alex → Tom (SERVICE: order tickets)
    links["alex_to_tom"] = Link(
        id="alex_to_tom",
        source_id="alex",
        dest_id="tom",
        link_type=LinkType.SERVICE,
        properties={
            "flow_types": ["order_ticket"],
            "bandwidth": 1.0,
        },
    )

    # Tom → Alex (SERVICE: finished sandwiches)
    links["tom_to_alex"] = Link(
        id="tom_to_alex",
        source_id="tom",
        dest_id="alex",
        link_type=LinkType.SERVICE,
        properties={
            "flow_types": ["finished_sandwich"],
            "bandwidth": 1.0,
        },
    )

    return links


def generate_customer_order_payload() -> dict[str, Any]:
    """Generate a random customer order payload.

    Returns:
        dict: Order details with sandwich type, extras, and special requests.
    """
    return {
        "sandwich_type": random.choice(SANDWICH_TYPES),
        "extras": random.sample(EXTRAS, k=random.randint(0, 2)),
        "special_requests": random.sample(SPECIAL_REQUESTS, k=random.randint(0, 1)),
    }


def customer_arrival_schedule(tick: int) -> float:
    """Return the rate multiplier for customer arrivals based on tick.

    Lunch rush doubles the rate between ticks 200-400.

    Args:
        tick: Current simulation tick.

    Returns:
        float: Rate multiplier (2.0 during lunch rush, 1.0 otherwise).
    """
    if 200 <= tick <= 400:
        return 2.0
    return 1.0


def create_external_inputs() -> list[ExternalInput]:
    """Create external inputs for the sandwich shop per PRD section 9.5.

    External inputs:
    - customer_arrivals: Customer orders targeting Alex at rate 0.05/tick

    Returns:
        list[ExternalInput]: External inputs for the world.
    """
    return [
        ExternalInput(
            name="customer_arrivals",
            target_agent_id="alex",
            rate=0.05,
            variance=0.3,
            particle_type="customer_order",
            payload_generator=generate_customer_order_payload,
            schedule=customer_arrival_schedule,
        ),
    ]


def create_labels() -> dict[str, Label]:
    """Create the 5 labels for the sandwich shop per PRD section 9.4.

    Labels:
    - SandwichShop: Global shop context with health_code, operating_hours, etc.
    - FrontLine: Customer-facing context with appearance and demeanor norms
    - Kitchen: Food preparation context with safety constraints and equipment
    - Management: Administrative context with supplier and scheduling resources
    - Register: Point-of-sale context with terminal and cash handling

    Returns:
        dict[str, Label]: Labels keyed by name.
    """
    labels = {}

    # SandwichShop: global context for the entire shop
    labels["SandwichShop"] = Label(
        name="SandwichShop",
        context=LabelContext(
            constraints=["health_code", "operating_hours_10_to_8", "max_capacity_30"],
            resources=["ingredient_supply", "POS_system", "kitchen_equipment"],
            norms=["FIFO_orders", "customer_greeting", "clean_as_you_go"],
        ),
    )

    # FrontLine: customer-facing roles
    labels["FrontLine"] = Label(
        name="FrontLine",
        context=LabelContext(
            constraints=["customer_facing_appearance"],
            norms=["friendly_demeanor", "no_phone_use"],
        ),
    )

    # Kitchen: food preparation area
    labels["Kitchen"] = Label(
        name="Kitchen",
        context=LabelContext(
            constraints=["food_safety_gloves", "temperature_monitoring"],
            resources=["grill", "prep_station", "cold_storage"],
        ),
    )

    # Management: administrative roles
    labels["Management"] = Label(
        name="Management",
        context=LabelContext(
            resources=["supplier_contacts", "POS_admin", "scheduling_system"],
            norms=["daily_inventory_check", "weekly_supply_order"],
        ),
    )

    # Register: point-of-sale area
    labels["Register"] = Label(
        name="Register",
        context=LabelContext(
            resources=["POS_terminal", "cash_drawer"],
            constraints=["cash_handling_policy"],
        ),
    )

    return labels


def create_world() -> World:
    """Create the sandwich shop world with links, labels, and external inputs.

    Returns:
        World: A world containing the sandwich shop links, labels, and external inputs.
    """
    world = World()
    world.links = create_links()
    world.labels = create_labels()
    world.external_inputs = create_external_inputs()
    return world

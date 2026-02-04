"""The Sandwich Shop corpus: A small sandwich shop with three employees.

This is the first implementation corpus for LoopEngine, demonstrating
agents, links, labels, and external inputs.
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


def maria_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """Maria's owner policy: observe shop state and issue directives or supply orders.

    On SENSE: observe shop state (queue depth, supply levels, throughput rate)
    On DECIDE: if queue depth exceeds threshold, send directive to Alex/Tom
               if supply levels low, generate supply order
               if healthy, do nothing (conserve attention)

    Args:
        sensed_inputs: Particles received (status_reports, revenue_reports, stockout_alerts)
        genome: Maria's genome traits
        internal_state: Current internal state

    Returns:
        list[Particle]: Output particles (directives, supply_orders)
    """
    outputs: list[Particle] = []

    # Process incoming reports
    for particle in sensed_inputs:
        if particle.particle_type == "stockout_alert":
            # Generate supply order if cost-sensitive threshold allows
            if genome.get("cost_sensitivity", 0.5) < 0.95:  # Very cost-sensitive might delay
                outputs.append(
                    Particle(
                        id=f"supply_order_{particle.id}",
                        particle_type="supply_order",
                        payload={"item": particle.payload.get("item", "general_supplies")},
                        source_id="maria",
                        dest_id="external",
                        link_id="",
                    )
                )

    # Check queue depth from internal state and issue directives
    queue_depth = internal_state.get("observed_queue_depth", 0)
    threshold = 5 * (1 - genome.get("decisiveness", 0.6))  # More decisive = lower threshold

    if queue_depth > threshold:
        # Issue directive based on delegation trait
        if genome.get("delegation", 0.7) > 0.5:
            outputs.append(
                Particle(
                    id=f"directive_speed_up_{internal_state.get('tick', 0)}",
                    particle_type="directive",
                    payload={"action": "speed_up", "urgency": "high"},
                    source_id="maria",
                    dest_id="alex",
                    link_id="maria_to_alex",
                )
            )

    return outputs


def tom_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """Tom's sandwich_maker policy: read tickets, make sandwiches, report status.

    On SENSE: read next ticket from input buffer
    On ORIENT: check ingredient availability
    On DECIDE: plan assembly or substitute based on ingredient_intuition
    On ACT: produce sandwich, send to Alex, report stockouts to Maria

    Args:
        sensed_inputs: Particles received (order_tickets, directives, ingredients)
        genome: Tom's genome traits
        internal_state: Current internal state

    Returns:
        list[Particle]: Output particles (finished_sandwich, status_report, stockout_alert)
    """
    outputs: list[Particle] = []

    # Initialize fitness tracking metrics if needed
    if "sandwiches_completed" not in internal_state:
        internal_state["sandwiches_completed"] = 0
    if "waste_count" not in internal_state:
        internal_state["waste_count"] = 0
    if "quality_scores" not in internal_state:
        internal_state["quality_scores"] = []
    if "ingredients_used" not in internal_state:
        internal_state["ingredients_used"] = 0

    for particle in sensed_inputs:
        if particle.particle_type == "order_ticket":
            # Process the order - quality based on consistency and speed tradeoff
            speed = genome.get("speed", 0.7)
            consistency = genome.get("consistency", 0.8)

            # Faster work may sacrifice consistency
            quality = consistency * (1 - (speed - 0.5) * 0.2)
            quality = max(0.5, min(1.0, quality))

            # Create finished sandwich
            outputs.append(
                Particle(
                    id=f"sandwich_{particle.id}",
                    particle_type="finished_sandwich",
                    payload={
                        "order": particle.payload,
                        "quality": quality,
                        "maker": "tom",
                    },
                    source_id="tom",
                    dest_id="alex",
                    link_id="tom_to_alex",
                )
            )

            # Track fitness metrics
            internal_state["sandwiches_completed"] += 1
            internal_state["quality_scores"].append(quality)
            internal_state["ingredients_used"] += 1  # Base ingredient per sandwich

            # Check for waste based on waste_minimization trait
            waste_chance = 0.1 * (1 - genome.get("waste_minimization", 0.5))
            if internal_state.get("random", lambda: 0.5)() < waste_chance:
                outputs.append(
                    Particle(
                        id=f"waste_{particle.id}",
                        particle_type="waste",
                        payload={"reason": "ingredient_spillage"},
                        source_id="tom",
                        dest_id="external",
                        link_id="",
                    )
                )
                # Track waste for fitness
                internal_state["waste_count"] += 1
                internal_state["ingredients_used"] += 1  # Wasted ingredient

        elif particle.particle_type == "directive":
            # Acknowledge directive
            internal_state["current_directive"] = particle.payload

    return outputs


def alex_policy(
    sensed_inputs: list[Particle],
    genome: dict[str, float],
    internal_state: dict[str, Any],
) -> list[Particle]:
    """Alex's cashier policy: take orders, create tickets, serve customers.

    On SENSE: check for waiting customer
    On ORIENT: read customer order
    On DECIDE: create order ticket
    On ACT: emit ticket to Tom, match sandwiches to customers, send revenue reports

    Args:
        sensed_inputs: Particles received (customer_order, finished_sandwich, directive)
        genome: Alex's genome traits
        internal_state: Current internal state

    Returns:
        list[Particle]: Output particles (order_ticket, served_customer, revenue_report)
    """
    outputs: list[Particle] = []

    # Initialize waiting customers queue in internal state if needed
    if "waiting_customers" not in internal_state:
        internal_state["waiting_customers"] = []
    if "max_queue_depth" not in internal_state:
        internal_state["max_queue_depth"] = 0

    for particle in sensed_inputs:
        if particle.particle_type == "customer_order":
            # Create order ticket for Tom
            outputs.append(
                Particle(
                    id=f"ticket_{particle.id}",
                    particle_type="order_ticket",
                    payload=particle.payload,
                    source_id="alex",
                    dest_id="tom",
                    link_id="alex_to_tom",
                )
            )
            # Track waiting customer
            internal_state["waiting_customers"].append(particle.id)
            # Track max queue depth for fitness evaluation
            current_depth = len(internal_state["waiting_customers"])
            if current_depth > internal_state["max_queue_depth"]:
                internal_state["max_queue_depth"] = current_depth

        elif particle.particle_type == "finished_sandwich":
            # Match sandwich to waiting customer
            if internal_state["waiting_customers"]:
                customer_id = internal_state["waiting_customers"].pop(0)
                # Serve customer
                outputs.append(
                    Particle(
                        id=f"served_{customer_id}",
                        particle_type="served_customer",
                        payload={
                            "customer_id": customer_id,
                            "sandwich": particle.payload,
                        },
                        source_id="alex",
                        dest_id="external",
                        link_id="",
                    )
                )

                # Periodic revenue report to Maria
                served_count = internal_state.get("served_count", 0) + 1
                internal_state["served_count"] = served_count
                if served_count % 10 == 0:
                    outputs.append(
                        Particle(
                            id=f"revenue_report_{served_count}",
                            particle_type="revenue_report",
                            payload={"total_served": served_count},
                            source_id="alex",
                            dest_id="maria",
                            link_id="alex_to_maria",
                        )
                    )

        elif particle.particle_type == "directive":
            # Store directive for behavior modification
            internal_state["current_directive"] = particle.payload

    return outputs


def create_agents() -> dict[str, Agent]:
    """Create the 3 agents for the sandwich shop per PRD section 9.2.

    Agents:
    - Maria (owner): loop_period=300, Management label, strategic decisions
    - Tom (sandwich_maker): loop_period=30, Kitchen label, fast production
    - Alex (cashier): loop_period=20, Register label, customer interaction

    Returns:
        dict[str, Agent]: Agents keyed by id.
    """
    agents = {}

    # Maria - Owner
    agents["maria"] = Agent(
        id="maria",
        name="Maria",
        role="owner",
        genome={
            "supply_forecasting": 0.7,
            "observation": 0.8,
            "decisiveness": 0.6,
            "delegation": 0.7,
            "cost_sensitivity": 0.9,
        },
        labels={"SandwichShop", "Management"},
        loop_period=300,
        policy=maria_policy,
    )

    # Tom - Sandwich Maker
    agents["tom"] = Agent(
        id="tom",
        name="Tom",
        role="sandwich_maker",
        genome={
            "speed": 0.7,
            "consistency": 0.8,
            "ingredient_intuition": 0.6,
            "stress_tolerance": 0.7,
            "waste_minimization": 0.5,
        },
        labels={"SandwichShop", "FrontLine", "Kitchen"},
        loop_period=30,
        policy=tom_policy,
    )

    # Alex - Cashier
    agents["alex"] = Agent(
        id="alex",
        name="Alex",
        role="cashier",
        genome={
            "speed": 0.8,
            "accuracy": 0.7,
            "friendliness": 0.8,
            "stress_tolerance": 0.6,
            "upselling": 0.5,
        },
        labels={"SandwichShop", "FrontLine", "Register"},
        loop_period=20,
        policy=alex_policy,
    )

    return agents


def create_world() -> World:
    """Create the sandwich shop world with agents, links, labels, and external inputs.

    Returns:
        World: A world containing the complete sandwich shop simulation.
    """
    world = World()
    world.agents = create_agents()
    world.links = create_links()
    world.labels = create_labels()
    world.external_inputs = create_external_inputs()
    return world

"""The Sandwich Shop corpus: A small sandwich shop with three employees.

This is the first implementation corpus for LoopEngine, demonstrating
agents, links, labels, and external inputs.
"""

from __future__ import annotations

from loopengine.model import Link, LinkType, World


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


def create_world() -> World:
    """Create the sandwich shop world with links.

    Returns:
        World: A world containing the sandwich shop links.
    """
    world = World()
    world.links = create_links()
    return world

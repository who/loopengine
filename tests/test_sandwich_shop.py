"""Tests for the Sandwich Shop corpus."""

from loopengine.corpora.sandwich_shop import create_links, create_world
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

"""Tests for the prompt builder module."""

import json

import pytest

from loopengine.behaviors.prompt_builder import (
    AgentContext,
    DomainContext,
    PromptBuilder,
)


@pytest.fixture
def builder() -> PromptBuilder:
    """Create a PromptBuilder instance."""
    return PromptBuilder()


@pytest.fixture
def flower_shop_domain() -> DomainContext:
    """Create a flower shop domain context."""
    return DomainContext(
        domain_type="flower shop",
        domain_description="A small florist shop that creates custom bouquets and arrangements",
    )


@pytest.fixture
def florist_agent() -> AgentContext:
    """Create a florist agent context."""
    return AgentContext(
        agent_type="florist",
        agent_role="Prepares flower arrangements and serves customers",
    )


@pytest.fixture
def full_context() -> dict:
    """Create a full simulation context."""
    return {
        "current_state": "idle",
        "pending_orders": 3,
        "available_flowers": ["roses", "tulips", "lilies"],
        "time_of_day": "morning",
        "energy_level": 0.8,
    }


class TestDomainContext:
    """Tests for DomainContext model."""

    def test_create_with_required_fields(self) -> None:
        """Test creating domain context with required fields only."""
        domain = DomainContext(domain_type="sandwich shop")
        assert domain.domain_type == "sandwich shop"
        assert domain.domain_description == ""

    def test_create_with_all_fields(self) -> None:
        """Test creating domain context with all fields."""
        domain = DomainContext(
            domain_type="warehouse",
            domain_description="A logistics warehouse with pickers and packers",
        )
        assert domain.domain_type == "warehouse"
        assert domain.domain_description == "A logistics warehouse with pickers and packers"


class TestAgentContext:
    """Tests for AgentContext model."""

    def test_create_with_required_fields(self) -> None:
        """Test creating agent context with required fields only."""
        agent = AgentContext(agent_type="worker")
        assert agent.agent_type == "worker"
        assert agent.agent_role == ""

    def test_create_with_all_fields(self) -> None:
        """Test creating agent context with all fields."""
        agent = AgentContext(
            agent_type="driver",
            agent_role="Delivers flowers to customers",
        )
        assert agent.agent_type == "driver"
        assert agent.agent_role == "Delivers flowers to customers"


class TestPromptBuilderBuildPrompt:
    """Tests for PromptBuilder.build_prompt()."""

    def test_prompt_contains_all_required_sections(
        self,
        builder: PromptBuilder,
        flower_shop_domain: DomainContext,
        florist_agent: AgentContext,
        full_context: dict,
    ) -> None:
        """Test prompt contains all required sections."""
        prompt = builder.build_prompt(flower_shop_domain, florist_agent, full_context)

        assert "Domain Description:" in prompt
        assert "Agent Type:" in prompt
        assert "Agent Role:" in prompt
        assert "Current Context:" in prompt
        assert "action:" in prompt
        assert "parameters:" in prompt
        assert "reasoning:" in prompt

    def test_prompt_with_minimal_context(self, builder: PromptBuilder) -> None:
        """Test prompt with minimal context - no errors."""
        domain = DomainContext(domain_type="generic")
        agent = AgentContext(agent_type="agent")

        prompt = builder.build_prompt(domain, agent, None)

        assert "Agent Type: agent" in prompt
        assert "Domain Description:" not in prompt
        assert "Agent Role:" not in prompt
        assert "Current Context:" not in prompt

    def test_prompt_with_full_context(
        self,
        builder: PromptBuilder,
        flower_shop_domain: DomainContext,
        florist_agent: AgentContext,
        full_context: dict,
    ) -> None:
        """Test prompt with full context - all fields present."""
        prompt = builder.build_prompt(flower_shop_domain, florist_agent, full_context)

        assert flower_shop_domain.domain_description in prompt
        assert florist_agent.agent_type in prompt
        assert florist_agent.agent_role in prompt
        assert "pending_orders" in prompt
        assert "morning" in prompt

    def test_context_json_is_valid(
        self,
        builder: PromptBuilder,
        flower_shop_domain: DomainContext,
        florist_agent: AgentContext,
        full_context: dict,
    ) -> None:
        """Test that JSON context in prompt is valid JSON."""
        prompt = builder.build_prompt(flower_shop_domain, florist_agent, full_context)

        context_start = prompt.find("Current Context:\n") + len("Current Context:\n")
        context_end = prompt.find("\n\nRespond with")
        context_json = prompt[context_start:context_end]

        parsed = json.loads(context_json)
        assert parsed["pending_orders"] == 3
        assert "roses" in parsed["available_flowers"]

    def test_handles_empty_context_dict(
        self,
        builder: PromptBuilder,
        flower_shop_domain: DomainContext,
        florist_agent: AgentContext,
    ) -> None:
        """Test prompt handles empty context dict."""
        prompt = builder.build_prompt(flower_shop_domain, florist_agent, {})

        assert "Current Context:" not in prompt

    def test_handles_complex_context_values(self, builder: PromptBuilder) -> None:
        """Test prompt handles complex nested context values."""
        domain = DomainContext(domain_type="test")
        agent = AgentContext(agent_type="test")
        context = {
            "nested": {"deep": {"value": 42}},
            "list_of_dicts": [{"a": 1}, {"b": 2}],
            "mixed": [1, "two", 3.0, None, True],
        }

        prompt = builder.build_prompt(domain, agent, context)

        context_start = prompt.find("Current Context:\n") + len("Current Context:\n")
        context_end = prompt.find("\n\nRespond with")
        context_json = prompt[context_start:context_end]

        parsed = json.loads(context_json)
        assert parsed["nested"]["deep"]["value"] == 42

    def test_handles_non_serializable_values(self, builder: PromptBuilder) -> None:
        """Test prompt handles non-JSON-serializable values via default=str."""
        from datetime import datetime

        domain = DomainContext(domain_type="test")
        agent = AgentContext(agent_type="test")
        dt = datetime(2024, 1, 15, 10, 30, 0)
        context = {"timestamp": dt}

        prompt = builder.build_prompt(domain, agent, context)

        assert "2024-01-15" in prompt


class TestPromptBuilderSystemMessage:
    """Tests for PromptBuilder.build_system_message()."""

    def test_system_message_includes_domain_type(
        self,
        builder: PromptBuilder,
        flower_shop_domain: DomainContext,
    ) -> None:
        """Test system message includes domain type."""
        message = builder.build_system_message(flower_shop_domain)

        assert "flower shop" in message
        assert "behavior engine" in message
        assert "simulation" in message

    def test_system_message_format(self, builder: PromptBuilder) -> None:
        """Test system message follows expected format."""
        domain = DomainContext(domain_type="warehouse")
        message = builder.build_system_message(domain)

        assert message.startswith("You are a behavior engine")
        assert "warehouse simulation" in message


class TestPromptBuilderFullPrompt:
    """Tests for PromptBuilder.build_full_prompt()."""

    def test_returns_tuple(
        self,
        builder: PromptBuilder,
        flower_shop_domain: DomainContext,
        florist_agent: AgentContext,
    ) -> None:
        """Test build_full_prompt returns tuple."""
        result = builder.build_full_prompt(flower_shop_domain, florist_agent, None)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_system_message_and_prompt_match_individual_methods(
        self,
        builder: PromptBuilder,
        flower_shop_domain: DomainContext,
        florist_agent: AgentContext,
        full_context: dict,
    ) -> None:
        """Test full prompt matches individual build methods."""
        system_message, prompt = builder.build_full_prompt(
            flower_shop_domain, florist_agent, full_context
        )

        expected_system = builder.build_system_message(flower_shop_domain)
        expected_prompt = builder.build_prompt(flower_shop_domain, florist_agent, full_context)

        assert system_message == expected_system
        assert prompt == expected_prompt


class TestPromptBuilderEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_domain_description(self, builder: PromptBuilder) -> None:
        """Test prompt with empty domain description omits that section."""
        domain = DomainContext(domain_type="test", domain_description="")
        agent = AgentContext(agent_type="agent")

        prompt = builder.build_prompt(domain, agent, None)

        assert "Domain Description:" not in prompt

    def test_empty_agent_role(self, builder: PromptBuilder) -> None:
        """Test prompt with empty agent role omits that section."""
        domain = DomainContext(domain_type="test")
        agent = AgentContext(agent_type="agent", agent_role="")

        prompt = builder.build_prompt(domain, agent, None)

        assert "Agent Role:" not in prompt

    def test_whitespace_in_domain_type(self, builder: PromptBuilder) -> None:
        """Test domain type with whitespace is handled correctly."""
        domain = DomainContext(domain_type="  flower shop  ")
        message = builder.build_system_message(domain)

        assert "  flower shop  " in message

    def test_special_characters_in_context(self, builder: PromptBuilder) -> None:
        """Test context with special characters is escaped properly in JSON."""
        domain = DomainContext(domain_type="test")
        agent = AgentContext(agent_type="agent")
        context = {
            "message": 'He said "hello"',
            "path": "C:\\Users\\test",
            "newline": "line1\nline2",
        }

        prompt = builder.build_prompt(domain, agent, context)

        context_start = prompt.find("Current Context:\n") + len("Current Context:\n")
        context_end = prompt.find("\n\nRespond with")
        context_json = prompt[context_start:context_end]

        parsed = json.loads(context_json)
        assert parsed["message"] == 'He said "hello"'
        assert parsed["path"] == "C:\\Users\\test"

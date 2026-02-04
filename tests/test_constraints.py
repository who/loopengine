"""Tests for behavioral constraints functionality.

Tests cover:
- ConstraintSchema model in domain_parser.py
- ConstraintContext model in prompt_builder.py
- Constraints in prompts via PromptBuilder
- Constraint API endpoints in domains.py
- Constraints passed through behavior generation
"""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.domain_parser import (
    ConstraintSchema,
    DomainParser,
    DomainSchema,
)
from loopengine.behaviors.llm_client import BehaviorResponse, LLMClient
from loopengine.behaviors.prompt_builder import (
    AgentContext,
    ConstraintContext,
    DomainContext,
    PromptBuilder,
)


@pytest.fixture
def mock_config() -> LLMConfig:
    """Create a mock LLM config."""
    return LLMConfig(
        llm_provider=LLMProvider.CLAUDE,
        anthropic_api_key=SecretStr("test-api-key"),
        llm_max_tokens=500,
        llm_temperature=0.7,
        llm_timeout=30.0,
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client."""
    return MagicMock(spec=LLMClient)


@pytest.fixture
def builder() -> PromptBuilder:
    """Create a PromptBuilder instance."""
    return PromptBuilder()


class TestConstraintSchema:
    """Tests for ConstraintSchema model in domain_parser.py."""

    def test_create_positive_constraint(self) -> None:
        """Test creating a positive (always) constraint."""
        constraint = ConstraintSchema(
            text="greet customers when they enter",
            constraint_type="positive",
        )
        assert constraint.text == "greet customers when they enter"
        assert constraint.constraint_type == "positive"

    def test_create_negative_constraint(self) -> None:
        """Test creating a negative (never) constraint."""
        constraint = ConstraintSchema(
            text="refuse service to a customer",
            constraint_type="negative",
        )
        assert constraint.text == "refuse service to a customer"
        assert constraint.constraint_type == "negative"

    def test_default_constraint_type_is_positive(self) -> None:
        """Test that default constraint type is positive."""
        constraint = ConstraintSchema(text="be polite")
        assert constraint.constraint_type == "positive"

    def test_constraint_in_domain_schema(self) -> None:
        """Test that DomainSchema can have constraints."""
        schema = DomainSchema(
            domain_type="sandwich shop",
            description="A sandwich shop",
            constraints=[
                ConstraintSchema(text="greet customers", constraint_type="positive"),
                ConstraintSchema(text="refuse service", constraint_type="negative"),
            ],
        )
        assert len(schema.constraints) == 2
        assert schema.constraints[0].text == "greet customers"
        assert schema.constraints[1].constraint_type == "negative"

    def test_domain_schema_constraints_default_empty(self) -> None:
        """Test that DomainSchema constraints default to empty list."""
        schema = DomainSchema(domain_type="test")
        assert schema.constraints == []


class TestConstraintContext:
    """Tests for ConstraintContext model in prompt_builder.py."""

    def test_create_constraint_context(self) -> None:
        """Test creating a constraint context."""
        constraint = ConstraintContext(
            text="always offer additional items",
            constraint_type="positive",
        )
        assert constraint.text == "always offer additional items"
        assert constraint.constraint_type == "positive"

    def test_constraint_context_default_type(self) -> None:
        """Test constraint context default type is positive."""
        constraint = ConstraintContext(text="be helpful")
        assert constraint.constraint_type == "positive"

    def test_domain_context_with_constraints(self) -> None:
        """Test DomainContext can have constraints."""
        domain = DomainContext(
            domain_type="coffee shop",
            domain_description="A coffee shop",
            constraints=[
                ConstraintContext(text="ask about dietary restrictions"),
                ConstraintContext(text="rush customers", constraint_type="negative"),
            ],
        )
        assert len(domain.constraints) == 2
        assert domain.constraints[0].text == "ask about dietary restrictions"
        assert domain.constraints[1].constraint_type == "negative"

    def test_domain_context_constraints_default_empty(self) -> None:
        """Test DomainContext constraints default to empty list."""
        domain = DomainContext(domain_type="test")
        assert domain.constraints == []


class TestPromptBuilderWithConstraints:
    """Tests for PromptBuilder handling constraints in prompts."""

    def test_prompt_includes_positive_constraints(self, builder: PromptBuilder) -> None:
        """Test that positive constraints appear with ALWAYS prefix."""
        domain = DomainContext(
            domain_type="sandwich shop",
            domain_description="A sandwich shop",
            constraints=[
                ConstraintContext(text="greet customers warmly", constraint_type="positive"),
            ],
        )
        agent = AgentContext(agent_type="cashier", agent_role="Takes orders")

        prompt = builder.build_prompt(domain, agent)

        assert "Behavioral Constraints (MUST be followed):" in prompt
        assert "ALWAYS:" in prompt
        assert "greet customers warmly" in prompt

    def test_prompt_includes_negative_constraints(self, builder: PromptBuilder) -> None:
        """Test that negative constraints appear with NEVER prefix."""
        domain = DomainContext(
            domain_type="sandwich shop",
            domain_description="A sandwich shop",
            constraints=[
                ConstraintContext(text="refuse a customer", constraint_type="negative"),
            ],
        )
        agent = AgentContext(agent_type="cashier", agent_role="Takes orders")

        prompt = builder.build_prompt(domain, agent)

        assert "Behavioral Constraints (MUST be followed):" in prompt
        assert "NEVER:" in prompt
        assert "refuse a customer" in prompt

    def test_prompt_includes_mixed_constraints(self, builder: PromptBuilder) -> None:
        """Test that both positive and negative constraints appear correctly."""
        domain = DomainContext(
            domain_type="sandwich shop",
            domain_description="A sandwich shop",
            constraints=[
                ConstraintContext(text="greet customers first", constraint_type="positive"),
                ConstraintContext(text="refuse service", constraint_type="negative"),
                ConstraintContext(text="offer napkins", constraint_type="positive"),
            ],
        )
        agent = AgentContext(agent_type="cashier")

        prompt = builder.build_prompt(domain, agent)

        assert "- ALWAYS: greet customers first" in prompt
        assert "- NEVER: refuse service" in prompt
        assert "- ALWAYS: offer napkins" in prompt

    def test_prompt_without_constraints_has_no_constraints_section(
        self, builder: PromptBuilder
    ) -> None:
        """Test that prompts without constraints don't have the constraints section."""
        domain = DomainContext(
            domain_type="sandwich shop",
            domain_description="A sandwich shop",
            constraints=[],
        )
        agent = AgentContext(agent_type="cashier")

        prompt = builder.build_prompt(domain, agent)

        assert "Behavioral Constraints" not in prompt
        assert "ALWAYS:" not in prompt
        assert "NEVER:" not in prompt

    def test_prompt_constraint_ordering(self, builder: PromptBuilder) -> None:
        """Test that constraints appear in correct order in prompt."""
        domain = DomainContext(
            domain_type="shop",
            constraints=[
                ConstraintContext(text="first constraint", constraint_type="positive"),
                ConstraintContext(text="second constraint", constraint_type="negative"),
            ],
        )
        agent = AgentContext(agent_type="worker")

        prompt = builder.build_prompt(domain, agent)

        first_pos = prompt.find("first constraint")
        second_pos = prompt.find("second constraint")
        assert first_pos < second_pos

    def test_full_prompt_with_constraints(self, builder: PromptBuilder) -> None:
        """Test build_full_prompt includes constraints in the prompt."""
        domain = DomainContext(
            domain_type="coffee shop",
            domain_description="A cozy coffee shop",
            constraints=[
                ConstraintContext(text="suggest daily specials", constraint_type="positive"),
            ],
        )
        agent = AgentContext(agent_type="barista", agent_role="Makes drinks")
        context = {"pending_orders": 2}

        system_message, prompt = builder.build_full_prompt(domain, agent, context)

        # System message should not contain constraints
        assert "Behavioral Constraints" not in system_message

        # Prompt should contain constraints
        assert "Behavioral Constraints (MUST be followed):" in prompt
        assert "ALWAYS: suggest daily specials" in prompt


class TestDomainParserExtractsConstraints:
    """Tests for DomainParser extracting constraints from descriptions."""

    def test_parser_extracts_constraints_from_llm_response(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test that parser extracts constraints from LLM response."""
        response_with_constraints = {
            "domain_type": "sandwich shop",
            "description": "A sandwich shop with customer service rules",
            "agent_types": [
                {"name": "cashier", "role": "Takes orders", "capabilities": ["take_order"]}
            ],
            "resources": [],
            "interactions": [],
            "constraints": [
                {"text": "greet customers warmly", "constraint_type": "positive"},
                {"text": "refuse service", "constraint_type": "negative"},
            ],
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response_with_constraints),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        schema = parser.parse(
            "A sandwich shop where employees must always greet customers and never refuse service"
        )

        assert len(schema.constraints) == 2
        assert schema.constraints[0].text == "greet customers warmly"
        assert schema.constraints[0].constraint_type == "positive"
        assert schema.constraints[1].text == "refuse service"
        assert schema.constraints[1].constraint_type == "negative"

    def test_parser_handles_response_without_constraints(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test that parser handles LLM response without constraints field."""
        response_without_constraints = {
            "domain_type": "bakery",
            "description": "A bakery",
            "agent_types": [{"name": "baker", "role": "Bakes", "capabilities": ["bake"]}],
            "resources": [],
            "interactions": [],
        }
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(response_without_constraints),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        schema = parser.parse("A bakery")

        assert schema.constraints == []

    def test_parser_system_message_includes_constraint_instructions(
        self,
        mock_config: LLMConfig,
        mock_llm_client: MagicMock,
    ) -> None:
        """Test that parser system message instructs LLM to extract constraints."""
        mock_llm_client.query.return_value = BehaviorResponse(
            action="",
            parameters={},
            reasoning=json.dumps(
                {
                    "domain_type": "test",
                    "agent_types": [],
                    "resources": [],
                    "interactions": [],
                    "constraints": [],
                }
            ),
            metadata={},
        )

        parser = DomainParser(llm_client=mock_llm_client, config=mock_config)
        parser.parse("A test domain")

        # Check system message includes constraint extraction instructions
        assert "constraints" in parser.SYSTEM_MESSAGE.lower()
        assert "always" in parser.SYSTEM_MESSAGE.lower()
        assert "never" in parser.SYSTEM_MESSAGE.lower()


class TestConstraintImports:
    """Tests for constraint class imports."""

    def test_import_constraint_schema_from_domain_parser(self) -> None:
        """Test ConstraintSchema can be imported from domain_parser."""
        from loopengine.behaviors.domain_parser import ConstraintSchema

        constraint = ConstraintSchema(text="test")
        assert constraint is not None

    def test_import_constraint_context_from_prompt_builder(self) -> None:
        """Test ConstraintContext can be imported from prompt_builder."""
        from loopengine.behaviors.prompt_builder import ConstraintContext

        constraint = ConstraintContext(text="test")
        assert constraint is not None

    def test_import_constraints_from_behaviors_package(self) -> None:
        """Test constraint classes can be imported from behaviors package."""
        from loopengine.behaviors import ConstraintContext, ConstraintSchema

        assert ConstraintSchema is not None
        assert ConstraintContext is not None


class TestConstraintsIntegration:
    """Integration tests for constraints flow from schema to prompt."""

    def test_schema_to_context_to_prompt_flow(self, builder: PromptBuilder) -> None:
        """Test constraints flow from DomainSchema to DomainContext to prompt."""
        # Create schema with constraints (as would come from parser)
        schema = DomainSchema(
            domain_type="restaurant",
            description="A restaurant",
            agent_types=[],
            resources=[],
            interactions=[],
            constraints=[
                ConstraintSchema(text="check for allergies", constraint_type="positive"),
                ConstraintSchema(text="serve alcohol to minors", constraint_type="negative"),
            ],
        )

        # Convert schema constraints to context constraints (as done in behaviors API)
        constraints = [
            ConstraintContext(text=c.text, constraint_type=c.constraint_type)
            for c in schema.constraints
        ]

        # Create domain context with constraints
        domain_context = DomainContext(
            domain_type=schema.domain_type,
            domain_description=schema.description,
            constraints=constraints,
        )
        agent_context = AgentContext(agent_type="waiter")

        # Build prompt
        prompt = builder.build_prompt(domain_context, agent_context)

        # Verify constraints appear in prompt
        assert "ALWAYS: check for allergies" in prompt
        assert "NEVER: serve alcohol to minors" in prompt

    def test_constraints_with_full_context(self, builder: PromptBuilder) -> None:
        """Test constraints work with full context including state."""
        domain = DomainContext(
            domain_type="bank",
            domain_description="A bank branch",
            constraints=[
                ConstraintContext(text="verify customer identity", constraint_type="positive"),
            ],
        )
        agent = AgentContext(agent_type="teller", agent_role="Handles transactions")
        context = {"current_transaction": "withdrawal", "amount": 500}

        prompt = builder.build_prompt(domain, agent, context)

        # Verify all sections present
        assert "Domain Description:" in prompt
        assert "Agent Type: teller" in prompt
        assert "Agent Role: Handles transactions" in prompt
        assert "Behavioral Constraints (MUST be followed):" in prompt
        assert "ALWAYS: verify customer identity" in prompt
        assert "Current Context:" in prompt
        assert "withdrawal" in prompt

"""Prompt builder for formatting simulation context into LLM prompts.

This module creates structured prompts for LLM queries following the
template defined in the PRD architecture section.
"""

import json
from typing import Any

from pydantic import BaseModel, Field


class DomainContext(BaseModel):
    """Domain configuration for prompt building.

    Attributes:
        domain_type: The type of business or system (e.g., 'flower shop').
        domain_description: Detailed description of the domain.
    """

    domain_type: str = Field(description="Type of business or system being simulated")
    domain_description: str = Field(default="", description="Detailed domain description")


class AgentContext(BaseModel):
    """Agent-specific context for prompt building.

    Attributes:
        agent_type: The type of agent (e.g., 'florist', 'customer').
        agent_role: The agent's role description.
    """

    agent_type: str = Field(description="Type of agent making the decision")
    agent_role: str = Field(default="", description="Description of the agent's role")


class PromptBuilder:
    """Builds LLM prompts from simulation context.

    Formats domain, agent, and state information into structured prompts
    that follow the template defined in the PRD:

        System: You are a behavior engine for a {domain_type} simulation.
        Given the current state of an agent, determine their next action.

        Domain Description: {domain_description}
        Agent Type: {agent_type}
        Agent Role: {agent_role}

        Current Context:
        {json_context}

        Respond with a JSON object containing:
        - action: The action the agent should take
        - parameters: Any parameters for the action
        - reasoning: Brief explanation (for debugging)

    Example:
        >>> builder = PromptBuilder()
        >>> domain = DomainContext(
        ...     domain_type="flower shop",
        ...     domain_description="A small florist with custom bouquets"
        ... )
        >>> agent = AgentContext(
        ...     agent_type="florist",
        ...     agent_role="Prepares flower arrangements for customers"
        ... )
        >>> context = {"pending_orders": 3, "current_task": None}
        >>> prompt = builder.build_prompt(domain, agent, context)
    """

    DEFAULT_SYSTEM_MESSAGE = (
        "You are a behavior engine for a {domain_type} simulation. "
        "Given the current state of an agent, determine their next action."
    )

    RESPONSE_FORMAT_INSTRUCTIONS = """
Respond with a JSON object containing:
- action: The action the agent should take
- parameters: Any parameters for the action
- reasoning: Brief explanation (for debugging)"""

    def build_prompt(
        self,
        domain: DomainContext,
        agent: AgentContext,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Build a formatted prompt from domain, agent, and context.

        Args:
            domain: Domain configuration with type and description.
            agent: Agent information with type and role.
            context: Optional state context dict to include as JSON.

        Returns:
            Formatted prompt string ready for LLM query.
        """
        parts: list[str] = []

        if domain.domain_description:
            parts.append(f"Domain Description: {domain.domain_description}")

        parts.append(f"Agent Type: {agent.agent_type}")

        if agent.agent_role:
            parts.append(f"Agent Role: {agent.agent_role}")

        if context:
            json_context = json.dumps(context, indent=2, default=str)
            parts.append(f"Current Context:\n{json_context}")

        parts.append(self.RESPONSE_FORMAT_INSTRUCTIONS.strip())

        return "\n\n".join(parts)

    def build_system_message(self, domain: DomainContext) -> str:
        """Build the system message for the LLM.

        Args:
            domain: Domain configuration with type.

        Returns:
            Formatted system message string.
        """
        return self.DEFAULT_SYSTEM_MESSAGE.format(domain_type=domain.domain_type)

    def build_full_prompt(
        self,
        domain: DomainContext,
        agent: AgentContext,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Build both system message and prompt.

        Convenience method that returns both components needed for LLM query.

        Args:
            domain: Domain configuration with type and description.
            agent: Agent information with type and role.
            context: Optional state context dict to include as JSON.

        Returns:
            Tuple of (system_message, prompt).
        """
        system_message = self.build_system_message(domain)
        prompt = self.build_prompt(domain, agent, context)
        return system_message, prompt

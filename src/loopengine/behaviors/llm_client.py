"""Abstract LLM client interface for behavior generation.

This module defines the base interface that all LLM providers must implement,
along with Pydantic models for request/response validation.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMClientSettings(BaseModel):
    """Configuration settings for an LLM client.

    Attributes:
        api_key: API key for the provider (optional for local models).
        model: Model identifier to use for queries.
        max_tokens: Maximum tokens in the response.
        temperature: Response randomness (0.0 to 1.0).
        timeout: Request timeout in seconds.
    """

    api_key: str | None = Field(default=None, description="API key for the provider")
    model: str = Field(default="", description="Model identifier")
    max_tokens: int = Field(default=500, ge=1, le=100000, description="Max response tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response randomness")
    timeout: float = Field(default=30.0, gt=0, description="Request timeout in seconds")


class LLMQuery(BaseModel):
    """Query to send to an LLM for behavior generation.

    Attributes:
        prompt: The formatted prompt string to send.
        context: Additional context data for the query.
        system_message: Optional system message to set LLM behavior.
    """

    prompt: str = Field(description="The formatted prompt string")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    system_message: str | None = Field(default=None, description="System message for LLM")


class BehaviorResponse(BaseModel):
    """Structured response from LLM containing agent behavior.

    Attributes:
        action: The action the agent should take.
        parameters: Parameters for the action.
        reasoning: Brief explanation of why this action was chosen.
        metadata: Additional metadata from the response.
    """

    action: str = Field(description="The action the agent should take")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the action"
    )
    reasoning: str = Field(default="", description="Explanation of why this action was chosen")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LLMClient(ABC):
    """Abstract base class for LLM provider clients.

    All LLM providers (Claude, OpenAI, Ollama) must implement this interface
    to be used by the behavior engine.

    Example:
        class ClaudeClient(LLMClient):
            def query(self, query: LLMQuery) -> BehaviorResponse:
                # Implementation using anthropic SDK
                ...

            def configure(self, settings: LLMClientSettings) -> None:
                # Apply settings
                ...
    """

    @abstractmethod
    def query(self, query: LLMQuery) -> BehaviorResponse:
        """Query the LLM for a behavior decision.

        Args:
            query: The query containing prompt and context.

        Returns:
            BehaviorResponse with the action, parameters, and reasoning.

        Raises:
            NotImplementedError: If not overridden by subclass.
            LLMClientError: If the query fails (timeout, rate limit, etc.).
        """
        raise NotImplementedError("Subclasses must implement query()")

    @abstractmethod
    def configure(self, settings: LLMClientSettings) -> None:
        """Configure the LLM client with new settings.

        Args:
            settings: The settings to apply.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement configure()")

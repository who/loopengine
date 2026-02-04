"""Claude (Anthropic) LLM client implementation.

This module provides the ClaudeClient class for querying Anthropic's Claude API
to generate agent behaviors.
"""

import json
import logging
from typing import Any

import anthropic
from anthropic import APIError, APITimeoutError, RateLimitError

from loopengine.behaviors.config import LLMConfig, get_llm_config
from loopengine.behaviors.llm_client import (
    BehaviorResponse,
    LLMClient,
    LLMClientSettings,
    LLMQuery,
)

logger = logging.getLogger(__name__)


class ClaudeClientError(Exception):
    """Exception raised when Claude API calls fail."""

    pass


class ClaudeClient(LLMClient):
    """LLM client for Anthropic's Claude API.

    Uses the official anthropic SDK to query Claude models for behavior generation.
    Handles retries on transient failures and returns structured JSON responses.

    Example:
        >>> client = ClaudeClient()
        >>> response = client.query(LLMQuery(
        ...     prompt="What action should this agent take?",
        ...     system_message="You are an agent behavior generator."
        ... ))
        >>> print(response.action)
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_RETRIES = 3

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize the Claude client.

        Args:
            config: Optional LLMConfig. If not provided, loads from environment.
        """
        self._config = config or get_llm_config()
        self._settings: LLMClientSettings | None = None
        self._client: anthropic.Anthropic | None = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Anthropic client with current configuration."""
        api_key = self._config.get_api_key()
        if not api_key:
            logger.warning("No Anthropic API key configured. Client will fail on query.")
            self._client = None
            return

        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=self._config.llm_timeout,
            max_retries=self.MAX_RETRIES,
        )

    def configure(self, settings: LLMClientSettings) -> None:
        """Configure the client with new settings.

        Args:
            settings: New settings to apply.
        """
        self._settings = settings

        # If settings include an API key, reinitialize the client
        if settings.api_key:
            self._client = anthropic.Anthropic(
                api_key=settings.api_key,
                timeout=settings.timeout,
                max_retries=self.MAX_RETRIES,
            )

    def query(self, query: LLMQuery) -> BehaviorResponse:
        """Query Claude for a behavior decision.

        Args:
            query: The query containing prompt and context.

        Returns:
            BehaviorResponse with the action, parameters, and reasoning.

        Raises:
            ClaudeClientError: If the query fails after retries.
        """
        if not self._client:
            raise ClaudeClientError(
                "Claude client not initialized. Check that ANTHROPIC_API_KEY is set."
            )

        # Determine settings to use
        model = self._get_model()
        max_tokens = self._get_max_tokens()
        temperature = self._get_temperature()

        # Build messages
        messages = self._build_messages(query)
        system_message = query.system_message or self._get_default_system_message()

        try:
            logger.debug("Querying Claude model %s with %d messages", model, len(messages))

            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=messages,
            )

            return self._parse_response(response)

        except APITimeoutError as e:
            logger.error("Claude API timeout after %s seconds", self._config.llm_timeout)
            raise ClaudeClientError(f"API request timed out: {e}") from e
        except RateLimitError as e:
            logger.error("Claude API rate limit exceeded")
            raise ClaudeClientError(f"Rate limit exceeded: {e}") from e
        except APIError as e:
            logger.error("Claude API error: %s", e.message)
            raise ClaudeClientError(f"API error: {e.message}") from e
        except Exception as e:
            logger.error("Unexpected error querying Claude: %s", str(e))
            raise ClaudeClientError(f"Unexpected error: {e}") from e

    def _get_model(self) -> str:
        """Get the model to use for queries."""
        if self._settings and self._settings.model:
            return self._settings.model
        return self.DEFAULT_MODEL

    def _get_max_tokens(self) -> int:
        """Get max tokens setting."""
        if self._settings:
            return self._settings.max_tokens
        return self._config.llm_max_tokens

    def _get_temperature(self) -> float:
        """Get temperature setting."""
        if self._settings:
            return self._settings.temperature
        return self._config.llm_temperature

    def _get_default_system_message(self) -> str:
        """Get the default system message for behavior generation."""
        return """You are an agent behavior generator for a simulation system.
Your task is to decide what action an agent should take given the current context.

You MUST respond with valid JSON in this exact format:
{
    "action": "action_name",
    "parameters": {"key": "value"},
    "reasoning": "Brief explanation"
}

Keep responses concise and focused on the action decision."""

    def _build_messages(self, query: LLMQuery) -> list[dict[str, Any]]:
        """Build the messages list for the API call."""
        content = query.prompt
        if query.context:
            content = f"{query.prompt}\n\nContext:\n{json.dumps(query.context, indent=2)}"

        return [{"role": "user", "content": content}]

    def _parse_response(self, response: anthropic.types.Message) -> BehaviorResponse:
        """Parse the Claude response into a BehaviorResponse.

        Args:
            response: Raw response from Claude API.

        Returns:
            Parsed BehaviorResponse.

        Raises:
            ClaudeClientError: If response cannot be parsed as valid JSON.
        """
        # Extract text content from response
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content = block.text
                break

        if not text_content:
            raise ClaudeClientError("Empty response from Claude API")

        # Try to parse as JSON
        try:
            # Handle potential markdown code blocks
            json_text = text_content
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()

            data = json.loads(json_text)

            return BehaviorResponse(
                action=data.get("action", "idle"),
                parameters=data.get("parameters", {}),
                reasoning=data.get("reasoning", ""),
                metadata={
                    "model": response.model,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "stop_reason": response.stop_reason,
                },
            )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Claude response as JSON: %s", text_content[:200])
            # Return a fallback response with raw text in reasoning
            return BehaviorResponse(
                action="idle",
                parameters={},
                reasoning=f"Failed to parse response: {text_content[:500]}",
                metadata={
                    "parse_error": str(e),
                    "raw_response": text_content[:1000],
                },
            )

"""OpenAI LLM client implementation.

This module provides the OpenAIClient class for querying OpenAI's API
to generate agent behaviors.
"""

import json
import logging
from typing import Any

import openai
from openai import APIError, APITimeoutError, RateLimitError

from loopengine.behaviors.config import LLMConfig, get_llm_config
from loopengine.behaviors.llm_client import (
    BehaviorResponse,
    LLMClient,
    LLMClientSettings,
    LLMQuery,
)

logger = logging.getLogger(__name__)


class OpenAIClientError(Exception):
    """Exception raised when OpenAI API calls fail."""

    pass


class OpenAIClient(LLMClient):
    """LLM client for OpenAI's API.

    Uses the official openai SDK to query OpenAI models for behavior generation.
    Handles retries on transient failures and returns structured JSON responses.

    Example:
        >>> client = OpenAIClient()
        >>> response = client.query(LLMQuery(
        ...     prompt="What action should this agent take?",
        ...     system_message="You are an agent behavior generator."
        ... ))
        >>> print(response.action)
    """

    DEFAULT_MODEL = "gpt-4"
    MAX_RETRIES = 3

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize the OpenAI client.

        Args:
            config: Optional LLMConfig. If not provided, loads from environment.
        """
        self._config = config or get_llm_config()
        self._settings: LLMClientSettings | None = None
        self._client: openai.OpenAI | None = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client with current configuration."""
        api_key = self._get_openai_api_key()
        if not api_key:
            logger.warning("No OpenAI API key configured. Client will fail on query.")
            self._client = None
            return

        self._client = openai.OpenAI(
            api_key=api_key,
            timeout=self._config.llm_timeout,
            max_retries=self.MAX_RETRIES,
        )

    def _get_openai_api_key(self) -> str | None:
        """Get the OpenAI API key from config."""
        if self._config.openai_api_key:
            return self._config.openai_api_key.get_secret_value()
        return None

    def configure(self, settings: LLMClientSettings) -> None:
        """Configure the client with new settings.

        Args:
            settings: New settings to apply.
        """
        self._settings = settings

        # If settings include an API key, reinitialize the client
        if settings.api_key:
            self._client = openai.OpenAI(
                api_key=settings.api_key,
                timeout=settings.timeout,
                max_retries=self.MAX_RETRIES,
            )

    def query(self, query: LLMQuery) -> BehaviorResponse:
        """Query OpenAI for a behavior decision.

        Args:
            query: The query containing prompt and context.

        Returns:
            BehaviorResponse with the action, parameters, and reasoning.

        Raises:
            OpenAIClientError: If the query fails after retries.
        """
        if not self._client:
            raise OpenAIClientError(
                "OpenAI client not initialized. Check that OPENAI_API_KEY is set."
            )

        # Determine settings to use
        model = self._get_model()
        max_tokens = self._get_max_tokens()
        temperature = self._get_temperature()

        # Build messages
        messages = self._build_messages(query)

        try:
            logger.debug("Querying OpenAI model %s with %d messages", model, len(messages))

            response = self._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )

            return self._parse_response(response)

        except APITimeoutError as e:
            logger.error("OpenAI API timeout after %s seconds", self._config.llm_timeout)
            raise OpenAIClientError(f"API request timed out: {e}") from e
        except RateLimitError as e:
            logger.error("OpenAI API rate limit exceeded")
            raise OpenAIClientError(f"Rate limit exceeded: {e}") from e
        except APIError as e:
            logger.error("OpenAI API error: %s", e.message)
            raise OpenAIClientError(f"API error: {e.message}") from e
        except Exception as e:
            logger.error("Unexpected error querying OpenAI: %s", str(e))
            raise OpenAIClientError(f"Unexpected error: {e}") from e

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
        system_message = query.system_message or self._get_default_system_message()

        content = query.prompt
        if query.context:
            content = f"{query.prompt}\n\nContext:\n{json.dumps(query.context, indent=2)}"

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": content},
        ]

    def _parse_response(self, response: openai.types.chat.ChatCompletion) -> BehaviorResponse:
        """Parse the OpenAI response into a BehaviorResponse.

        Args:
            response: Raw response from OpenAI API.

        Returns:
            Parsed BehaviorResponse.

        Raises:
            OpenAIClientError: If response cannot be parsed as valid JSON.
        """
        # Extract text content from response
        if not response.choices:
            raise OpenAIClientError("Empty response from OpenAI API")

        choice = response.choices[0]
        if not choice.message or not choice.message.content:
            raise OpenAIClientError("Empty message content from OpenAI API")

        text_content = choice.message.content

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
                        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    },
                    "finish_reason": choice.finish_reason,
                },
            )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse OpenAI response as JSON: %s", text_content[:200])
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

"""Ollama local LLM client implementation.

This module provides the OllamaClient class for querying local Ollama models
to generate agent behaviors. Ollama enables offline/free LLM usage with
models running locally.
"""

import json
import logging
from typing import Any

import httpx

from loopengine.behaviors.config import LLMConfig, get_llm_config
from loopengine.behaviors.llm_client import (
    BehaviorResponse,
    LLMClient,
    LLMClientSettings,
    LLMQuery,
)

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Exception raised when Ollama API calls fail."""

    pass


class OllamaClient(LLMClient):
    """LLM client for local Ollama models.

    Uses HTTP requests to query Ollama's REST API for behavior generation.
    Handles connection failures gracefully and returns structured JSON responses.

    Example:
        >>> client = OllamaClient()
        >>> response = client.query(LLMQuery(
        ...     prompt="What action should this agent take?",
        ...     system_message="You are an agent behavior generator."
        ... ))
        >>> print(response.action)
    """

    DEFAULT_MODEL = "llama2"

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize the Ollama client.

        Args:
            config: Optional LLMConfig. If not provided, loads from environment.
        """
        self._config = config or get_llm_config()
        self._settings: LLMClientSettings | None = None
        self._host = self._config.ollama_host
        self._model = self._config.ollama_model

    def configure(self, settings: LLMClientSettings) -> None:
        """Configure the client with new settings.

        Args:
            settings: New settings to apply.
        """
        self._settings = settings

        # Settings can override the model
        if settings.model:
            self._model = settings.model

    def query(self, query: LLMQuery) -> BehaviorResponse:
        """Query Ollama for a behavior decision.

        Args:
            query: The query containing prompt and context.

        Returns:
            BehaviorResponse with the action, parameters, and reasoning.

        Raises:
            OllamaClientError: If the query fails (connection, timeout, etc.).
        """
        model = self._get_model()
        timeout = self._get_timeout()

        # Build the prompt with system message and context
        prompt = self._build_prompt(query)

        try:
            logger.debug("Querying Ollama model %s at %s", model, self._host)

            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{self._host}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self._get_temperature(),
                            "num_predict": self._get_max_tokens(),
                        },
                    },
                )
                response.raise_for_status()

            return self._parse_response(response.json())

        except httpx.ConnectError as e:
            logger.error("Failed to connect to Ollama at %s: %s", self._host, str(e))
            raise OllamaClientError(
                f"Cannot connect to Ollama at {self._host}. "
                "Ensure Ollama is running (ollama serve)."
            ) from e
        except httpx.TimeoutException as e:
            logger.error("Ollama request timed out after %s seconds", timeout)
            raise OllamaClientError(f"Request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error("Ollama HTTP error: %s", e.response.status_code)
            raise OllamaClientError(f"HTTP error: {e.response.status_code}") from e
        except Exception as e:
            logger.error("Unexpected error querying Ollama: %s", str(e))
            raise OllamaClientError(f"Unexpected error: {e}") from e

    def _get_model(self) -> str:
        """Get the model to use for queries."""
        if self._settings and self._settings.model:
            return self._settings.model
        return self._model or self.DEFAULT_MODEL

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

    def _get_timeout(self) -> float:
        """Get timeout setting."""
        if self._settings:
            return self._settings.timeout
        return self._config.llm_timeout

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

    def _build_prompt(self, query: LLMQuery) -> str:
        """Build the complete prompt including system message and context."""
        system_message = query.system_message or self._get_default_system_message()

        content = query.prompt
        if query.context:
            content = f"{query.prompt}\n\nContext:\n{json.dumps(query.context, indent=2)}"

        # Ollama uses a single prompt, so we combine system and user messages
        return f"System: {system_message}\n\nUser: {content}"

    def _parse_response(self, response_data: dict[str, Any]) -> BehaviorResponse:
        """Parse the Ollama response into a BehaviorResponse.

        Args:
            response_data: Raw JSON response from Ollama API.

        Returns:
            Parsed BehaviorResponse.
        """
        text_content = response_data.get("response", "")

        if not text_content:
            raise OllamaClientError("Empty response from Ollama API")

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
                    "model": response_data.get("model", self._model),
                    "usage": {
                        "total_duration": response_data.get("total_duration", 0),
                        "eval_count": response_data.get("eval_count", 0),
                        "prompt_eval_count": response_data.get("prompt_eval_count", 0),
                    },
                    "done": response_data.get("done", True),
                },
            )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Ollama response as JSON: %s", text_content[:200])
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

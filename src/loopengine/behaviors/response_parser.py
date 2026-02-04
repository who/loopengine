"""Response parser for extracting structured actions from LLM responses.

This module provides the ResponseParser class for parsing raw LLM text responses
into validated BehaviorResponse objects with robust error handling.
"""

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from loopengine.behaviors.llm_client import BehaviorResponse

logger = logging.getLogger(__name__)


class ResponseParserError(Exception):
    """Exception raised when response parsing fails."""

    pass


class ResponseParser:
    """Parser for extracting structured behavior data from LLM responses.

    Handles various LLM response formats including:
    - Raw JSON
    - JSON wrapped in markdown code blocks (```json ... ```)
    - JSON with surrounding text

    Example:
        >>> parser = ResponseParser()
        >>> response = parser.parse('{"action": "move", "parameters": {"x": 10}}')
        >>> print(response.action)
        'move'
    """

    # Default action when parsing fails and fallback is enabled
    DEFAULT_ACTION = "idle"

    def __init__(self, strict: bool = False) -> None:
        """Initialize the response parser.

        Args:
            strict: If True, raise errors on parse failures. If False, return
                   fallback response with error details in metadata.
        """
        self._strict = strict

    def parse(self, llm_response: str) -> BehaviorResponse:
        """Parse an LLM response into a BehaviorResponse.

        Args:
            llm_response: Raw text response from an LLM.

        Returns:
            BehaviorResponse with extracted action, parameters, and reasoning.

        Raises:
            ResponseParserError: If strict mode and parsing fails.
        """
        if not llm_response or not llm_response.strip():
            return self._handle_error("Empty response from LLM", llm_response)

        # Extract JSON from the response
        json_text = self._extract_json(llm_response)

        try:
            data = json.loads(json_text)
            return self._build_response(data, llm_response)
        except json.JSONDecodeError as e:
            return self._handle_error(f"Invalid JSON: {e}", llm_response)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from potentially wrapped text.

        Handles:
        - Raw JSON
        - ```json ... ``` blocks
        - ``` ... ``` blocks
        - JSON embedded in surrounding text

        Args:
            text: Raw text that may contain JSON.

        Returns:
            Extracted JSON string.
        """
        # Try markdown json code block first
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Try generic code block
        if "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                # Verify it looks like JSON
                if extracted.startswith("{") or extracted.startswith("["):
                    return extracted

        # Try to find JSON object by matching balanced braces
        json_str = self._find_balanced_json(text, "{", "}")
        if json_str:
            return json_str

        # Return as-is and let JSON parser handle errors
        return text.strip()

    def _find_balanced_json(self, text: str, open_char: str, close_char: str) -> str | None:
        """Find balanced JSON structure in text.

        Args:
            text: Text to search.
            open_char: Opening character ('{' or '[').
            close_char: Closing character ('}' or ']').

        Returns:
            Extracted JSON string or None if not found.
        """
        start = text.find(open_char)
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i, char in enumerate(text[start:], start):
            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"' and not escape:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None

    def _build_response(self, data: dict[str, Any], raw_response: str) -> BehaviorResponse:
        """Build a BehaviorResponse from parsed data.

        Args:
            data: Parsed JSON data.
            raw_response: Original raw response for error context.

        Returns:
            Validated BehaviorResponse.

        Raises:
            ResponseParserError: If strict mode and validation fails.
        """
        if not isinstance(data, dict):
            return self._handle_error(
                f"Expected JSON object, got {type(data).__name__}", raw_response
            )

        # Check for required 'action' field
        if "action" not in data:
            return self._handle_error("Missing required field: action", raw_response)

        try:
            return BehaviorResponse(
                action=str(data["action"]),
                parameters=data.get("parameters", {}),
                reasoning=str(data.get("reasoning", "")),
                metadata=self._build_metadata(data),
            )
        except ValidationError as e:
            return self._handle_error(f"Validation error: {e}", raw_response)

    def _build_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Build metadata from extra fields in the response.

        Args:
            data: Parsed JSON data.

        Returns:
            Metadata dictionary with any extra fields.
        """
        standard_fields = {"action", "parameters", "reasoning"}
        extra_fields = {k: v for k, v in data.items() if k not in standard_fields}

        if extra_fields:
            return {"extra_fields": extra_fields}
        return {}

    def _handle_error(self, error_message: str, raw_response: str) -> BehaviorResponse:
        """Handle parsing errors based on strict mode.

        Args:
            error_message: Description of the error.
            raw_response: Original raw response.

        Returns:
            Fallback BehaviorResponse if not strict.

        Raises:
            ResponseParserError: If strict mode is enabled.
        """
        logger.warning("Response parsing failed: %s", error_message)

        if self._strict:
            raise ResponseParserError(error_message)

        # Return fallback response with error details
        return BehaviorResponse(
            action=self.DEFAULT_ACTION,
            parameters={},
            reasoning=f"Parse failed: {error_message}",
            metadata={
                "parse_error": error_message,
                "raw_response": raw_response[:1000] if raw_response else "",
            },
        )

    def parse_with_retry_hint(self, llm_response: str) -> tuple[BehaviorResponse, str | None]:
        """Parse response and generate a retry hint if parsing fails.

        This method is useful when the caller wants to ask the LLM to
        fix a malformed response.

        Args:
            llm_response: Raw text response from an LLM.

        Returns:
            Tuple of (BehaviorResponse, retry_hint). retry_hint is None if
            parsing succeeded, otherwise contains a prompt for the LLM to
            fix its response.
        """
        if not llm_response or not llm_response.strip():
            hint = (
                "Your previous response was empty. Please provide a valid JSON "
                'response with this format: {"action": "...", "parameters": {...}, '
                '"reasoning": "..."}'
            )
            return self._handle_error("Empty response", llm_response), hint

        json_text = self._extract_json(llm_response)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            hint = (
                f"Your previous response was not valid JSON (error: {e}). "
                "Please respond with ONLY valid JSON in this format: "
                '{"action": "...", "parameters": {...}, "reasoning": "..."}'
            )
            return self._handle_error(f"Invalid JSON: {e}", llm_response), hint

        if not isinstance(data, dict):
            hint = (
                f"Your previous response was a {type(data).__name__}, not a JSON object. "
                "Please respond with a JSON object: "
                '{"action": "...", "parameters": {...}, "reasoning": "..."}'
            )
            return self._handle_error(
                f"Expected object, got {type(data).__name__}", llm_response
            ), hint

        if "action" not in data:
            hint = (
                'Your previous response was missing the required "action" field. '
                "Please include it: "
                '{"action": "your_action_here", "parameters": {...}, "reasoning": "..."}'
            )
            return self._handle_error("Missing action field", llm_response), hint

        # Parsing succeeded
        try:
            response = BehaviorResponse(
                action=str(data["action"]),
                parameters=data.get("parameters", {}),
                reasoning=str(data.get("reasoning", "")),
                metadata=self._build_metadata(data),
            )
            return response, None
        except ValidationError as e:
            hint = f"Your response had validation errors: {e}. Please fix and retry."
            return self._handle_error(f"Validation error: {e}", llm_response), hint

"""Tests for the ResponseParser class."""

import pytest

from loopengine.behaviors import BehaviorResponse
from loopengine.behaviors.response_parser import ResponseParser, ResponseParserError


class TestResponseParserBasicParsing:
    """Tests for basic JSON parsing functionality."""

    def test_parse_valid_json(self) -> None:
        """Test parsing a valid JSON response."""
        parser = ResponseParser()
        response = parser.parse(
            '{"action": "move", "parameters": {"x": 10, "y": 20}, "reasoning": "Moving to target"}'
        )

        assert response.action == "move"
        assert response.parameters == {"x": 10, "y": 20}
        assert response.reasoning == "Moving to target"

    def test_parse_minimal_json(self) -> None:
        """Test parsing JSON with only required action field."""
        parser = ResponseParser()
        response = parser.parse('{"action": "wait"}')

        assert response.action == "wait"
        assert response.parameters == {}
        assert response.reasoning == ""

    def test_parse_json_with_extra_fields(self) -> None:
        """Test extra fields are captured in metadata."""
        parser = ResponseParser()
        response = parser.parse(
            '{"action": "attack", "parameters": {"target": "enemy"}, '
            '"confidence": 0.95, "alternative": "flee"}'
        )

        assert response.action == "attack"
        assert response.parameters == {"target": "enemy"}
        assert "extra_fields" in response.metadata
        assert response.metadata["extra_fields"]["confidence"] == 0.95
        assert response.metadata["extra_fields"]["alternative"] == "flee"

    def test_parse_nested_parameters(self) -> None:
        """Test parsing JSON with nested parameters."""
        parser = ResponseParser()
        json_str = (
            '{"action": "craft", "parameters": '
            '{"recipe": {"input": ["wood", "iron"], "output": "sword"}}}'
        )
        response = parser.parse(json_str)

        assert response.action == "craft"
        assert response.parameters["recipe"]["input"] == ["wood", "iron"]
        assert response.parameters["recipe"]["output"] == "sword"


class TestResponseParserCodeBlocks:
    """Tests for parsing JSON from markdown code blocks."""

    def test_parse_json_code_block(self) -> None:
        """Test parsing JSON wrapped in ```json block."""
        parser = ResponseParser()
        json_str = (
            '```json\n{"action": "harvest", "parameters": {"resource": "wheat"}, '
            '"reasoning": "Crops ready"}\n```'
        )
        response = parser.parse(json_str)

        assert response.action == "harvest"
        assert response.parameters == {"resource": "wheat"}
        assert response.reasoning == "Crops ready"

    def test_parse_generic_code_block(self) -> None:
        """Test parsing JSON wrapped in generic ``` block."""
        parser = ResponseParser()
        response = parser.parse(
            '```\n{"action": "rest", "parameters": {}, "reasoning": "Low energy"}\n```'
        )

        assert response.action == "rest"
        assert response.reasoning == "Low energy"

    def test_parse_json_with_surrounding_text(self) -> None:
        """Test parsing JSON embedded in surrounding text."""
        parser = ResponseParser()
        text = (
            "Based on the situation, I recommend: "
            '{"action": "defend", "parameters": {"position": "north"}} '
            "This will protect the base."
        )
        response = parser.parse(text)

        assert response.action == "defend"
        assert response.parameters == {"position": "north"}

    def test_parse_json_with_whitespace(self) -> None:
        """Test parsing JSON with extra whitespace."""
        parser = ResponseParser()
        response = parser.parse('  ```json  \n  {"action": "scout", "parameters": {}}  \n  ```  ')

        assert response.action == "scout"


class TestResponseParserErrorHandling:
    """Tests for error handling and fallback behavior."""

    def test_parse_empty_response(self) -> None:
        """Test empty response returns fallback action."""
        parser = ResponseParser()
        response = parser.parse("")

        assert response.action == "idle"
        assert "parse_error" in response.metadata
        assert "Empty response" in response.metadata["parse_error"]

    def test_parse_whitespace_only(self) -> None:
        """Test whitespace-only response returns fallback."""
        parser = ResponseParser()
        response = parser.parse("   \n\t  ")

        assert response.action == "idle"
        assert "parse_error" in response.metadata

    def test_parse_invalid_json(self) -> None:
        """Test invalid JSON returns fallback action."""
        parser = ResponseParser()
        response = parser.parse("This is not JSON at all")

        assert response.action == "idle"
        assert "parse_error" in response.metadata
        assert "Invalid JSON" in response.metadata["parse_error"]
        assert "raw_response" in response.metadata

    def test_parse_malformed_json(self) -> None:
        """Test malformed JSON returns fallback action."""
        parser = ResponseParser()
        response = parser.parse('{"action": "move", "parameters":}')

        assert response.action == "idle"
        assert "parse_error" in response.metadata

    def test_parse_missing_action_field(self) -> None:
        """Test JSON without action field returns fallback."""
        parser = ResponseParser()
        response = parser.parse('{"parameters": {"x": 10}, "reasoning": "Going there"}')

        assert response.action == "idle"
        assert "parse_error" in response.metadata
        assert "action" in response.metadata["parse_error"]

    def test_parse_json_array_extracts_object(self) -> None:
        """Test JSON array with object extracts the inner object."""
        parser = ResponseParser()
        # Parser finds the first JSON object { } in the text
        response = parser.parse('[{"action": "move"}]')

        # Parser extracts the inner object since it looks for balanced {}
        assert response.action == "move"

    def test_parse_pure_json_array_returns_fallback(self) -> None:
        """Test pure JSON array (no object) returns fallback."""
        parser = ResponseParser()
        response = parser.parse('["move", "wait", "idle"]')

        assert response.action == "idle"
        assert "parse_error" in response.metadata


class TestResponseParserStrictMode:
    """Tests for strict mode error handling."""

    def test_strict_mode_empty_raises_error(self) -> None:
        """Test strict mode raises error on empty response."""
        parser = ResponseParser(strict=True)

        with pytest.raises(ResponseParserError) as exc_info:
            parser.parse("")

        assert "Empty response" in str(exc_info.value)

    def test_strict_mode_invalid_json_raises_error(self) -> None:
        """Test strict mode raises error on invalid JSON."""
        parser = ResponseParser(strict=True)

        with pytest.raises(ResponseParserError) as exc_info:
            parser.parse("not json")

        assert "Invalid JSON" in str(exc_info.value)

    def test_strict_mode_missing_action_raises_error(self) -> None:
        """Test strict mode raises error when action missing."""
        parser = ResponseParser(strict=True)

        with pytest.raises(ResponseParserError) as exc_info:
            parser.parse('{"parameters": {}}')

        assert "action" in str(exc_info.value)

    def test_strict_mode_valid_json_succeeds(self) -> None:
        """Test strict mode succeeds with valid JSON."""
        parser = ResponseParser(strict=True)
        response = parser.parse('{"action": "proceed"}')

        assert response.action == "proceed"


class TestResponseParserRetryHint:
    """Tests for parse_with_retry_hint method."""

    def test_retry_hint_on_empty_response(self) -> None:
        """Test retry hint generated for empty response."""
        parser = ResponseParser()
        response, hint = parser.parse_with_retry_hint("")

        assert response.action == "idle"
        assert hint is not None
        assert "empty" in hint.lower()
        assert "JSON" in hint

    def test_retry_hint_on_invalid_json(self) -> None:
        """Test retry hint generated for invalid JSON."""
        parser = ResponseParser()
        response, hint = parser.parse_with_retry_hint("not valid json")

        assert response.action == "idle"
        assert hint is not None
        assert "not valid JSON" in hint

    def test_retry_hint_on_missing_action(self) -> None:
        """Test retry hint generated for missing action."""
        parser = ResponseParser()
        response, hint = parser.parse_with_retry_hint('{"parameters": {}}')

        assert response.action == "idle"
        assert hint is not None
        assert "action" in hint

    def test_retry_hint_on_json_array(self) -> None:
        """Test retry hint generated for JSON array."""
        parser = ResponseParser()
        response, hint = parser.parse_with_retry_hint("[]")

        assert response.action == "idle"
        assert hint is not None
        assert "object" in hint

    def test_no_retry_hint_on_success(self) -> None:
        """Test no retry hint on successful parse."""
        parser = ResponseParser()
        response, hint = parser.parse_with_retry_hint('{"action": "continue"}')

        assert response.action == "continue"
        assert hint is None


class TestResponseParserImport:
    """Tests for import and instantiation."""

    def test_import_from_behaviors(self) -> None:
        """Test ResponseParser can be imported from behaviors package."""
        from loopengine.behaviors import ResponseParser, ResponseParserError

        assert ResponseParser is not None
        assert ResponseParserError is not None

    def test_returns_behavior_response(self) -> None:
        """Test parse returns BehaviorResponse instance."""
        parser = ResponseParser()
        response = parser.parse('{"action": "test"}')

        assert isinstance(response, BehaviorResponse)


class TestResponseParserEdgeCases:
    """Tests for edge cases and complex scenarios."""

    def test_parse_with_unicode(self) -> None:
        """Test parsing JSON with unicode characters."""
        parser = ResponseParser()
        json_str = (
            '{"action": "speak", "parameters": {"message": "Hola, ¿cómo estás?"}, '
            '"reasoning": "👋 Greeting"}'
        )
        response = parser.parse(json_str)

        assert response.action == "speak"
        assert "Hola" in response.parameters["message"]

    def test_parse_with_escaped_strings(self) -> None:
        """Test parsing JSON with escaped characters."""
        parser = ResponseParser()
        response = parser.parse(
            '{"action": "log", "parameters": {"message": "Line1\\nLine2\\tTab"}}'
        )

        assert response.action == "log"
        assert "Line1\nLine2\tTab" == response.parameters["message"]

    def test_parse_numeric_action_converted_to_string(self) -> None:
        """Test numeric action is converted to string."""
        parser = ResponseParser()
        response = parser.parse('{"action": 42, "parameters": {}}')

        assert response.action == "42"
        assert isinstance(response.action, str)

    def test_parse_boolean_action_converted_to_string(self) -> None:
        """Test boolean action is converted to string."""
        parser = ResponseParser()
        response = parser.parse('{"action": true, "parameters": {}}')

        assert response.action == "True"
        assert isinstance(response.action, str)

    def test_parse_deeply_nested_code_blocks(self) -> None:
        """Test parsing handles multiple code blocks."""
        parser = ResponseParser()
        # Should extract from the first valid json block
        response = parser.parse(
            'Here is my response:\n```json\n{"action": "first"}\n```\n'
            'And another example:\n```json\n{"action": "second"}\n```'
        )

        assert response.action == "first"

    def test_parse_truncates_long_raw_response_in_metadata(self) -> None:
        """Test long raw responses are truncated in metadata."""
        parser = ResponseParser()
        long_response = "x" * 2000  # 2000 character invalid response

        response = parser.parse(long_response)

        assert response.action == "idle"
        assert len(response.metadata["raw_response"]) == 1000

    def test_parse_empty_parameters_object(self) -> None:
        """Test empty parameters object is handled."""
        parser = ResponseParser()
        response = parser.parse('{"action": "idle", "parameters": {}}')

        assert response.action == "idle"
        assert response.parameters == {}

    def test_parse_null_reasoning(self) -> None:
        """Test null reasoning is handled."""
        parser = ResponseParser()
        response = parser.parse('{"action": "idle", "reasoning": null}')

        assert response.action == "idle"
        assert response.reasoning == "None"

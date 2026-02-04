"""Tests for the Claude LLM client implementation."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.llm_client import LLMClientSettings, LLMQuery
from loopengine.behaviors.providers.claude import ClaudeClient, ClaudeClientError


@pytest.fixture
def mock_config() -> LLMConfig:
    """Create a mock LLM config with test API key."""
    return LLMConfig(
        llm_provider=LLMProvider.CLAUDE,
        anthropic_api_key=SecretStr("test-api-key"),
        llm_max_tokens=500,
        llm_temperature=0.7,
        llm_timeout=30.0,
    )


@pytest.fixture
def mock_anthropic_response() -> MagicMock:
    """Create a mock Anthropic API response."""
    response = MagicMock()
    json_response = '{"action": "move", "parameters": {"x": 10, "y": 20}, "reasoning": "Moving"}'
    response.content = [MagicMock(type="text", text=json_response)]
    response.model = "claude-sonnet-4-20250514"
    response.usage = MagicMock(input_tokens=100, output_tokens=50)
    response.stop_reason = "end_turn"
    return response


class TestClaudeClientInit:
    """Tests for ClaudeClient initialization."""

    def test_init_with_config(self, mock_config: LLMConfig) -> None:
        """Test client initializes with provided config."""
        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            client = ClaudeClient(config=mock_config)
            assert client._config == mock_config
            mock_anthropic.assert_called_once()

    def test_init_without_api_key_logs_warning(self) -> None:
        """Test client logs warning when no API key is set."""
        config = LLMConfig(
            llm_provider=LLMProvider.CLAUDE,
            anthropic_api_key=None,
        )
        with patch("loopengine.behaviors.providers.claude.logger") as mock_logger:
            client = ClaudeClient(config=config)
            mock_logger.warning.assert_called_once()
            assert client._client is None


class TestClaudeClientConfigure:
    """Tests for ClaudeClient.configure()."""

    def test_configure_updates_settings(self, mock_config: LLMConfig) -> None:
        """Test configure updates internal settings."""
        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic"):
            client = ClaudeClient(config=mock_config)

            settings = LLMClientSettings(
                api_key="new-key",
                model="claude-opus-4-20250514",
                max_tokens=1000,
                temperature=0.5,
            )
            client.configure(settings)

            assert client._settings == settings

    def test_configure_with_api_key_reinitializes_client(self, mock_config: LLMConfig) -> None:
        """Test configure reinitializes client when new API key provided."""
        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            client = ClaudeClient(config=mock_config)

            settings = LLMClientSettings(api_key="new-key")
            client.configure(settings)

            # Should have been called twice: once at init, once at configure
            assert mock_anthropic.call_count == 2


class TestClaudeClientQuery:
    """Tests for ClaudeClient.query()."""

    def test_query_success(
        self, mock_config: LLMConfig, mock_anthropic_response: MagicMock
    ) -> None:
        """Test successful query returns BehaviorResponse."""
        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(config=mock_config)
            query = LLMQuery(prompt="What action should the agent take?")

            response = client.query(query)

            assert response.action == "move"
            assert response.parameters == {"x": 10, "y": 20}
            assert response.reasoning == "Moving"
            assert "model" in response.metadata
            assert "usage" in response.metadata

    def test_query_with_context(
        self, mock_config: LLMConfig, mock_anthropic_response: MagicMock
    ) -> None:
        """Test query includes context in prompt."""
        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(config=mock_config)
            query = LLMQuery(
                prompt="What action?",
                context={"agent_type": "worker", "energy": 50},
                system_message="You are a test agent.",
            )

            client.query(query)

            # Verify context was included in the message
            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            assert "Context" in messages[0]["content"]
            assert "agent_type" in messages[0]["content"]

    def test_query_uses_configured_settings(
        self, mock_config: LLMConfig, mock_anthropic_response: MagicMock
    ) -> None:
        """Test query uses settings from configure()."""
        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(config=mock_config)
            client.configure(
                LLMClientSettings(
                    model="custom-model",
                    max_tokens=2000,
                    temperature=0.3,
                )
            )

            query = LLMQuery(prompt="Test")
            client.query(query)

            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["model"] == "custom-model"
            assert call_args.kwargs["max_tokens"] == 2000
            assert call_args.kwargs["temperature"] == 0.3

    def test_query_without_client_raises_error(self) -> None:
        """Test query raises error when client not initialized."""
        config = LLMConfig(
            llm_provider=LLMProvider.CLAUDE,
            anthropic_api_key=None,
        )
        client = ClaudeClient(config=config)

        with pytest.raises(ClaudeClientError) as exc_info:
            client.query(LLMQuery(prompt="Test"))

        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_query_handles_timeout(self, mock_config: LLMConfig) -> None:
        """Test query handles timeout errors gracefully."""
        from anthropic import APITimeoutError

        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = APITimeoutError(request=MagicMock())
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(config=mock_config)

            with pytest.raises(ClaudeClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "timed out" in str(exc_info.value)

    def test_query_handles_rate_limit(self, mock_config: LLMConfig) -> None:
        """Test query handles rate limit errors gracefully."""
        from anthropic import RateLimitError

        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(config=mock_config)

            with pytest.raises(ClaudeClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "Rate limit" in str(exc_info.value)

    def test_query_handles_api_error(self, mock_config: LLMConfig) -> None:
        """Test query handles generic API errors gracefully."""
        from anthropic import APIError

        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = APIError(
                message="Internal server error",
                request=MagicMock(),
                body=None,
            )
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(config=mock_config)

            with pytest.raises(ClaudeClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "API error" in str(exc_info.value)


class TestClaudeClientResponseParsing:
    """Tests for response parsing logic."""

    def test_parse_json_with_code_block(self, mock_config: LLMConfig) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        response = MagicMock()
        response.content = [
            MagicMock(
                type="text",
                text='```json\n{"action": "wait", "parameters": {}, "reasoning": "Waiting"}\n```',
            )
        ]
        response.model = "claude-sonnet-4-20250514"
        response.usage = MagicMock(input_tokens=10, output_tokens=20)
        response.stop_reason = "end_turn"

        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(config=mock_config)
            result = client.query(LLMQuery(prompt="Test"))

            assert result.action == "wait"

    def test_parse_invalid_json_returns_fallback(self, mock_config: LLMConfig) -> None:
        """Test invalid JSON returns fallback response."""
        response = MagicMock()
        response.content = [MagicMock(type="text", text="This is not JSON")]
        response.model = "claude-sonnet-4-20250514"
        response.usage = MagicMock(input_tokens=10, output_tokens=20)
        response.stop_reason = "end_turn"

        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(config=mock_config)
            result = client.query(LLMQuery(prompt="Test"))

            assert result.action == "idle"
            assert "parse_error" in result.metadata

    def test_parse_empty_response_raises_error(self, mock_config: LLMConfig) -> None:
        """Test empty response raises error."""
        response = MagicMock()
        response.content = []

        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(config=mock_config)

            with pytest.raises(ClaudeClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "Empty response" in str(exc_info.value)


class TestClaudeClientImport:
    """Tests for import and instantiation."""

    def test_import_from_providers(self) -> None:
        """Test ClaudeClient can be imported from providers package."""
        from loopengine.behaviors.providers import ClaudeClient, ClaudeClientError

        assert ClaudeClient is not None
        assert ClaudeClientError is not None

    def test_implements_llm_client_interface(self, mock_config: LLMConfig) -> None:
        """Test ClaudeClient implements LLMClient interface."""
        from loopengine.behaviors.llm_client import LLMClient

        with patch("loopengine.behaviors.providers.claude.anthropic.Anthropic"):
            client = ClaudeClient(config=mock_config)
            assert isinstance(client, LLMClient)

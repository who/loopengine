"""Tests for the OpenAI LLM client implementation."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.llm_client import LLMClientSettings, LLMQuery
from loopengine.behaviors.providers.openai import OpenAIClient, OpenAIClientError


@pytest.fixture
def mock_config() -> LLMConfig:
    """Create a mock LLM config with test API key."""
    return LLMConfig(
        llm_provider=LLMProvider.OPENAI,
        openai_api_key=SecretStr("test-api-key"),
        llm_max_tokens=500,
        llm_temperature=0.7,
        llm_timeout=30.0,
    )


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """Create a mock OpenAI API response."""
    response = MagicMock()
    json_response = '{"action": "move", "parameters": {"x": 10, "y": 20}, "reasoning": "Moving"}'

    # Set up the message content
    message = MagicMock()
    message.content = json_response

    # Set up the choice
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    # Set up the usage
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50

    # Set up the response
    response.choices = [choice]
    response.model = "gpt-4"
    response.usage = usage

    return response


class TestOpenAIClientInit:
    """Tests for OpenAIClient initialization."""

    def test_init_with_config(self, mock_config: LLMConfig) -> None:
        """Test client initializes with provided config."""
        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            client = OpenAIClient(config=mock_config)
            assert client._config == mock_config
            mock_openai.assert_called_once()

    def test_init_without_api_key_logs_warning(self) -> None:
        """Test client logs warning when no API key is set."""
        config = LLMConfig(
            llm_provider=LLMProvider.OPENAI,
            openai_api_key=None,
        )
        with patch("loopengine.behaviors.providers.openai.logger") as mock_logger:
            client = OpenAIClient(config=config)
            mock_logger.warning.assert_called_once()
            assert client._client is None


class TestOpenAIClientConfigure:
    """Tests for OpenAIClient.configure()."""

    def test_configure_updates_settings(self, mock_config: LLMConfig) -> None:
        """Test configure updates internal settings."""
        with patch("loopengine.behaviors.providers.openai.openai.OpenAI"):
            client = OpenAIClient(config=mock_config)

            settings = LLMClientSettings(
                api_key="new-key",
                model="gpt-4-turbo",
                max_tokens=1000,
                temperature=0.5,
            )
            client.configure(settings)

            assert client._settings == settings

    def test_configure_with_api_key_reinitializes_client(self, mock_config: LLMConfig) -> None:
        """Test configure reinitializes client when new API key provided."""
        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            client = OpenAIClient(config=mock_config)

            settings = LLMClientSettings(api_key="new-key")
            client.configure(settings)

            # Should have been called twice: once at init, once at configure
            assert mock_openai.call_count == 2


class TestOpenAIClientQuery:
    """Tests for OpenAIClient.query()."""

    def test_query_success(self, mock_config: LLMConfig, mock_openai_response: MagicMock) -> None:
        """Test successful query returns BehaviorResponse."""
        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)
            query = LLMQuery(prompt="What action should the agent take?")

            response = client.query(query)

            assert response.action == "move"
            assert response.parameters == {"x": 10, "y": 20}
            assert response.reasoning == "Moving"
            assert "model" in response.metadata
            assert "usage" in response.metadata

    def test_query_with_context(
        self, mock_config: LLMConfig, mock_openai_response: MagicMock
    ) -> None:
        """Test query includes context in prompt."""
        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)
            query = LLMQuery(
                prompt="What action?",
                context={"agent_type": "worker", "energy": 50},
                system_message="You are a test agent.",
            )

            client.query(query)

            # Verify context was included in the message
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]

            # Check system message is custom
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a test agent."

            # Check user message contains context
            assert messages[1]["role"] == "user"
            assert "Context" in messages[1]["content"]
            assert "agent_type" in messages[1]["content"]

    def test_query_uses_configured_settings(
        self, mock_config: LLMConfig, mock_openai_response: MagicMock
    ) -> None:
        """Test query uses settings from configure()."""
        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)
            client.configure(
                LLMClientSettings(
                    model="gpt-4-turbo",
                    max_tokens=2000,
                    temperature=0.3,
                )
            )

            query = LLMQuery(prompt="Test")
            client.query(query)

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "gpt-4-turbo"
            assert call_args.kwargs["max_tokens"] == 2000
            assert call_args.kwargs["temperature"] == 0.3

    def test_query_without_client_raises_error(self) -> None:
        """Test query raises error when client not initialized."""
        config = LLMConfig(
            llm_provider=LLMProvider.OPENAI,
            openai_api_key=None,
        )
        client = OpenAIClient(config=config)

        with pytest.raises(OpenAIClientError) as exc_info:
            client.query(LLMQuery(prompt="Test"))

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_query_handles_timeout(self, mock_config: LLMConfig) -> None:
        """Test query handles timeout errors gracefully."""
        from openai import APITimeoutError

        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)

            with pytest.raises(OpenAIClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "timed out" in str(exc_info.value)

    def test_query_handles_rate_limit(self, mock_config: LLMConfig) -> None:
        """Test query handles rate limit errors gracefully."""
        from openai import RateLimitError

        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)

            with pytest.raises(OpenAIClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "Rate limit" in str(exc_info.value)

    def test_query_handles_api_error(self, mock_config: LLMConfig) -> None:
        """Test query handles generic API errors gracefully."""
        from openai import APIError

        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = APIError(
                message="Internal server error",
                request=MagicMock(),
                body=None,
            )
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)

            with pytest.raises(OpenAIClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "API error" in str(exc_info.value)


class TestOpenAIClientResponseParsing:
    """Tests for response parsing logic."""

    def test_parse_json_with_code_block(self, mock_config: LLMConfig) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        response = MagicMock()
        message = MagicMock()
        message.content = (
            '```json\n{"action": "wait", "parameters": {}, "reasoning": "Waiting"}\n```'
        )
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)
            result = client.query(LLMQuery(prompt="Test"))

            assert result.action == "wait"

    def test_parse_invalid_json_returns_fallback(self, mock_config: LLMConfig) -> None:
        """Test invalid JSON returns fallback response."""
        response = MagicMock()
        message = MagicMock()
        message.content = "This is not JSON"
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)
            result = client.query(LLMQuery(prompt="Test"))

            assert result.action == "idle"
            assert "parse_error" in result.metadata

    def test_parse_empty_choices_raises_error(self, mock_config: LLMConfig) -> None:
        """Test empty choices raises error."""
        response = MagicMock()
        response.choices = []

        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)

            with pytest.raises(OpenAIClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "Empty response" in str(exc_info.value)

    def test_parse_empty_message_content_raises_error(self, mock_config: LLMConfig) -> None:
        """Test empty message content raises error."""
        response = MagicMock()
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        response.choices = [choice]

        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)

            with pytest.raises(OpenAIClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "Empty message content" in str(exc_info.value)


class TestOpenAIClientImport:
    """Tests for import and instantiation."""

    def test_import_from_providers(self) -> None:
        """Test OpenAIClient can be imported from providers package."""
        from loopengine.behaviors.providers import OpenAIClient, OpenAIClientError

        assert OpenAIClient is not None
        assert OpenAIClientError is not None

    def test_implements_llm_client_interface(self, mock_config: LLMConfig) -> None:
        """Test OpenAIClient implements LLMClient interface."""
        from loopengine.behaviors.llm_client import LLMClient

        with patch("loopengine.behaviors.providers.openai.openai.OpenAI"):
            client = OpenAIClient(config=mock_config)
            assert isinstance(client, LLMClient)


class TestOpenAIClientResponseStructure:
    """Tests to verify response structure matches Claude client output."""

    def test_response_metadata_structure(
        self, mock_config: LLMConfig, mock_openai_response: MagicMock
    ) -> None:
        """Test response metadata has compatible structure with Claude client."""
        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)
            response = client.query(LLMQuery(prompt="Test"))

            # Verify metadata structure matches Claude client
            assert "model" in response.metadata
            assert "usage" in response.metadata
            assert "input_tokens" in response.metadata["usage"]
            assert "output_tokens" in response.metadata["usage"]
            # OpenAI uses finish_reason instead of stop_reason
            assert "finish_reason" in response.metadata

    def test_behavior_response_fields(
        self, mock_config: LLMConfig, mock_openai_response: MagicMock
    ) -> None:
        """Test BehaviorResponse has all required fields."""
        with patch("loopengine.behaviors.providers.openai.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client

            client = OpenAIClient(config=mock_config)
            response = client.query(LLMQuery(prompt="Test"))

            # All required fields should be present
            assert hasattr(response, "action")
            assert hasattr(response, "parameters")
            assert hasattr(response, "reasoning")
            assert hasattr(response, "metadata")

            # Fields should have correct types
            assert isinstance(response.action, str)
            assert isinstance(response.parameters, dict)
            assert isinstance(response.reasoning, str)
            assert isinstance(response.metadata, dict)

"""Tests for the Ollama LLM client implementation."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from loopengine.behaviors.config import LLMConfig, LLMProvider
from loopengine.behaviors.llm_client import LLMClientSettings, LLMQuery
from loopengine.behaviors.providers.ollama import OllamaClient, OllamaClientError


@pytest.fixture
def mock_config() -> LLMConfig:
    """Create a mock LLM config for Ollama."""
    return LLMConfig(
        llm_provider=LLMProvider.OLLAMA,
        ollama_host="http://localhost:11434",
        ollama_model="llama2",
        llm_max_tokens=500,
        llm_temperature=0.7,
        llm_timeout=30.0,
    )


@pytest.fixture
def mock_ollama_response() -> dict:
    """Create a mock Ollama API response."""
    return {
        "model": "llama2",
        "response": '{"action": "move", "parameters": {"x": 10, "y": 20}, "reasoning": "Moving"}',
        "done": True,
        "total_duration": 1000000000,
        "eval_count": 50,
        "prompt_eval_count": 100,
    }


class TestOllamaClientInit:
    """Tests for OllamaClient initialization."""

    def test_init_with_config(self, mock_config: LLMConfig) -> None:
        """Test client initializes with provided config."""
        client = OllamaClient(config=mock_config)
        assert client._config == mock_config
        assert client._host == "http://localhost:11434"
        assert client._model == "llama2"

    def test_init_with_default_config(self) -> None:
        """Test client initializes with default config."""
        with patch("loopengine.behaviors.providers.ollama.get_llm_config") as mock_get_config:
            mock_get_config.return_value = LLMConfig(
                llm_provider=LLMProvider.OLLAMA,
                ollama_host="http://custom:11434",
                ollama_model="mistral",
            )
            client = OllamaClient()
            assert client._host == "http://custom:11434"
            assert client._model == "mistral"


class TestOllamaClientConfigure:
    """Tests for OllamaClient.configure()."""

    def test_configure_updates_settings(self, mock_config: LLMConfig) -> None:
        """Test configure updates internal settings."""
        client = OllamaClient(config=mock_config)

        settings = LLMClientSettings(
            model="codellama",
            max_tokens=1000,
            temperature=0.5,
            timeout=60.0,
        )
        client.configure(settings)

        assert client._settings == settings
        assert client._model == "codellama"

    def test_configure_without_model_keeps_original(self, mock_config: LLMConfig) -> None:
        """Test configure without model keeps original model."""
        client = OllamaClient(config=mock_config)
        original_model = client._model

        settings = LLMClientSettings(
            max_tokens=1000,
            temperature=0.5,
        )
        client.configure(settings)

        # Model should be unchanged
        assert client._model == original_model


class TestOllamaClientQuery:
    """Tests for OllamaClient.query()."""

    def test_query_success(self, mock_config: LLMConfig, mock_ollama_response: dict) -> None:
        """Test successful query returns BehaviorResponse."""
        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_ollama_response
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__enter__.return_value.post.return_value = mock_response

            query = LLMQuery(prompt="What action should the agent take?")
            response = client.query(query)

            assert response.action == "move"
            assert response.parameters == {"x": 10, "y": 20}
            assert response.reasoning == "Moving"
            assert response.metadata["model"] == "llama2"

    def test_query_with_context(self, mock_config: LLMConfig, mock_ollama_response: dict) -> None:
        """Test query includes context in prompt."""
        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_ollama_response
            mock_response.raise_for_status = MagicMock()
            mock_client = mock_httpx.return_value.__enter__.return_value
            mock_client.post.return_value = mock_response

            query = LLMQuery(
                prompt="What action?",
                context={"agent_type": "worker", "energy": 50},
                system_message="You are a test agent.",
            )

            client.query(query)

            # Verify context was included in the request
            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            assert "Context" in request_body["prompt"]
            assert "agent_type" in request_body["prompt"]

    def test_query_uses_configured_settings(
        self, mock_config: LLMConfig, mock_ollama_response: dict
    ) -> None:
        """Test query uses settings from configure()."""
        client = OllamaClient(config=mock_config)
        client.configure(
            LLMClientSettings(
                model="codellama",
                max_tokens=2000,
                temperature=0.3,
                timeout=60.0,
            )
        )

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_ollama_response
            mock_response.raise_for_status = MagicMock()
            mock_client = mock_httpx.return_value.__enter__.return_value
            mock_client.post.return_value = mock_response

            query = LLMQuery(prompt="Test")
            client.query(query)

            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            assert request_body["model"] == "codellama"
            assert request_body["options"]["num_predict"] == 2000
            assert request_body["options"]["temperature"] == 0.3

    def test_query_handles_connection_error(self, mock_config: LLMConfig) -> None:
        """Test query handles connection errors gracefully."""
        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError(
                "Connection refused"
            )

            with pytest.raises(OllamaClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "Cannot connect to Ollama" in str(exc_info.value)
            assert "ollama serve" in str(exc_info.value)

    def test_query_handles_timeout(self, mock_config: LLMConfig) -> None:
        """Test query handles timeout errors gracefully."""
        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value.post.side_effect = (
                httpx.TimeoutException("Request timed out")
            )

            with pytest.raises(OllamaClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "timed out" in str(exc_info.value)

    def test_query_handles_http_error(self, mock_config: LLMConfig) -> None:
        """Test query handles HTTP errors gracefully."""
        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Internal Server Error",
                request=MagicMock(),
                response=mock_response,
            )
            mock_httpx.return_value.__enter__.return_value.post.return_value = mock_response

            with pytest.raises(OllamaClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "HTTP error" in str(exc_info.value)


class TestOllamaClientResponseParsing:
    """Tests for response parsing logic."""

    def test_parse_json_with_code_block(self, mock_config: LLMConfig) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        json_content = '{"action": "wait", "parameters": {}, "reasoning": "Waiting"}'
        response_data = {
            "model": "llama2",
            "response": f"```json\n{json_content}\n```",
            "done": True,
        }

        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__enter__.return_value.post.return_value = mock_response

            result = client.query(LLMQuery(prompt="Test"))

            assert result.action == "wait"

    def test_parse_invalid_json_returns_fallback(self, mock_config: LLMConfig) -> None:
        """Test invalid JSON returns fallback response."""
        response_data = {
            "model": "llama2",
            "response": "This is not JSON",
            "done": True,
        }

        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__enter__.return_value.post.return_value = mock_response

            result = client.query(LLMQuery(prompt="Test"))

            assert result.action == "idle"
            assert "parse_error" in result.metadata

    def test_parse_empty_response_raises_error(self, mock_config: LLMConfig) -> None:
        """Test empty response raises error."""
        response_data = {
            "model": "llama2",
            "response": "",
            "done": True,
        }

        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__enter__.return_value.post.return_value = mock_response

            with pytest.raises(OllamaClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            assert "Empty response" in str(exc_info.value)

    def test_parse_response_includes_metadata(self, mock_config: LLMConfig) -> None:
        """Test parsed response includes Ollama metadata."""
        response_data = {
            "model": "llama2",
            "response": '{"action": "test", "parameters": {}, "reasoning": "Testing"}',
            "done": True,
            "total_duration": 2000000000,
            "eval_count": 100,
            "prompt_eval_count": 50,
        }

        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_response = MagicMock()
            mock_response.json.return_value = response_data
            mock_response.raise_for_status = MagicMock()
            mock_httpx.return_value.__enter__.return_value.post.return_value = mock_response

            result = client.query(LLMQuery(prompt="Test"))

            assert result.metadata["model"] == "llama2"
            assert result.metadata["usage"]["total_duration"] == 2000000000
            assert result.metadata["usage"]["eval_count"] == 100


class TestOllamaClientImport:
    """Tests for import and instantiation."""

    def test_import_from_providers(self) -> None:
        """Test OllamaClient can be imported from providers package."""
        from loopengine.behaviors.providers import OllamaClient, OllamaClientError

        assert OllamaClient is not None
        assert OllamaClientError is not None

    def test_implements_llm_client_interface(self, mock_config: LLMConfig) -> None:
        """Test OllamaClient implements LLMClient interface."""
        from loopengine.behaviors.llm_client import LLMClient

        client = OllamaClient(config=mock_config)
        assert isinstance(client, LLMClient)


class TestOllamaClientGracefulFailure:
    """Tests for graceful failure when Ollama is not available."""

    def test_not_running_gives_helpful_error(self, mock_config: LLMConfig) -> None:
        """Test helpful error message when Ollama is not running."""
        client = OllamaClient(config=mock_config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError(
                "Connection refused"
            )

            with pytest.raises(OllamaClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            error_msg = str(exc_info.value)
            assert "http://localhost:11434" in error_msg
            assert "ollama serve" in error_msg

    def test_wrong_host_gives_helpful_error(self) -> None:
        """Test helpful error message when wrong host is configured."""
        config = LLMConfig(
            llm_provider=LLMProvider.OLLAMA,
            ollama_host="http://wrong-host:11434",
            ollama_model="llama2",
        )
        client = OllamaClient(config=config)

        with patch("loopengine.behaviors.providers.ollama.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError(
                "Connection refused"
            )

            with pytest.raises(OllamaClientError) as exc_info:
                client.query(LLMQuery(prompt="Test"))

            error_msg = str(exc_info.value)
            assert "http://wrong-host:11434" in error_msg

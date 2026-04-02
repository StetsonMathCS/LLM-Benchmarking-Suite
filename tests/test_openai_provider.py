"""
End-to-end tests for the OpenAI provider.
Uses mocking to avoid real API calls.
"""
import pytest
from unittest.mock import patch, MagicMock
from core.base import ModelConfig, LLMResponse


@pytest.fixture
def config():
    return ModelConfig(
        provider="openai",
        model_name="gpt-4",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        temperature="0.7",
        max_tokens="1024",
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def mock_openai():
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


# ── Connection ────────────────────────────────────────────────────────────

class TestOpenAIConnection:
    def test_connect_succeeds(self, config, mock_openai):
        from providers.openai.provider import OpenAIProvider
        provider = OpenAIProvider(config)
        result = provider.connect()
        assert result is True

    def test_connect_without_openai_raises(self, config):
        with patch.dict("sys.modules", {"openai": None}):
            from providers.openai.provider import OpenAIProvider
            provider = OpenAIProvider(config)
            with pytest.raises((RuntimeError, ImportError)):
                provider.connect()


# ── Completion ────────────────────────────────────────────────────────────

class TestOpenAICompletion:
    def test_successful_completion(self, config, mock_openai):
        from providers.openai.provider import OpenAIProvider

        # Mock response
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello, World!"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_openai.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(config)
        provider.connect()
        response = provider.complete("Say hello")

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello, World!"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.error is None
        assert response.success is True

    def test_completion_with_system_prompt(self, config, mock_openai):
        from providers.openai.provider import OpenAIProvider

        mock_choice = MagicMock()
        mock_choice.message.content = "result"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 3
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_openai.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(config)
        provider.connect()
        provider.complete("Do something", system_prompt="Be concise")

        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise"

    def test_completion_error_returns_error_response(self, config, mock_openai):
        from providers.openai.provider import OpenAIProvider

        mock_openai.chat.completions.create.side_effect = Exception("API error")

        provider = OpenAIProvider(config)
        provider.connect()
        response = provider.complete("Say hello")

        assert isinstance(response, LLMResponse)
        assert response.error is not None
        assert "API error" in response.error
        assert response.success is False

    def test_temperature_retry(self, config, mock_openai):
        from providers.openai.provider import OpenAIProvider

        # First call fails with temperature error, second succeeds
        mock_choice = MagicMock()
        mock_choice.message.content = "retried"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 3
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_openai.chat.completions.create.side_effect = [
            Exception("temperature does not support this value"),
            mock_response,
        ]

        provider = OpenAIProvider(config)
        provider.connect()
        response = provider.complete("Say hello")

        assert response.content == "retried"
        assert response.error is None


# ── Availability ──────────────────────────────────────────────────────────

class TestOpenAIAvailability:
    def test_available_when_models_list_works(self, config, mock_openai):
        from providers.openai.provider import OpenAIProvider
        mock_openai.models.list.return_value = ["model1"]

        provider = OpenAIProvider(config)
        provider.connect()
        assert provider.is_available() is True

    def test_unavailable_when_models_list_fails(self, config, mock_openai):
        from providers.openai.provider import OpenAIProvider
        mock_openai.models.list.side_effect = Exception("connection error")

        provider = OpenAIProvider(config)
        provider.connect()
        assert provider.is_available() is False


# ── Repr ──────────────────────────────────────────────────────────────────

class TestOpenAIRepr:
    def test_repr(self, config, mock_openai):
        from providers.openai.provider import OpenAIProvider
        provider = OpenAIProvider(config)
        assert "gpt-4" in repr(provider)

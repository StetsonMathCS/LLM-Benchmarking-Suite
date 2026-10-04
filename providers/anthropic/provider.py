"""
providers/antropic/provider.py
Anthropic API provider
"""
from typing import Optional
from core.registry import ProviderRegistry
from core.base import BaseProvider, LLMResponse, usage_fields
import asyncio
from time import perf_counter

MESSAGE_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cache_creation",
    "server_tool_use",
)

@ProviderRegistry.register("anthropic")
class AnthropicProvider(BaseProvider):

    def connect(self) -> bool:
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key = self.config.api_key,
                max_retries=0,
                timeout=self.config.extra_params.get("_provider_timeout_s", 180),
            )
            return True
        except ImportError:
            raise RuntimeError("anthropic package is not installed. Run: pip install anthropic")
        
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        provider_extras = dict(self.config.extra_params)
        provider_extras.pop("_provider_timeout_s", None)
        kwargs = dict(
            model = self.config.model_name,
            max_tokens = self.config.max_tokens,
            messages = [{"role":"user", "content":prompt}],
            **provider_extras,
        )
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if system_prompt:
            kwargs["system"] = system_prompt
        try:
            started = perf_counter()
            response = self._client.messages.create(**kwargs)
            content = "".join(
                str(getattr(block, "text", ""))
                for block in response.content
                if getattr(block, "type", None) == "text" or hasattr(block, "text")
            )
            return LLMResponse(
                content=content,
                model=self.config.model_name,
                provider="anthropic",
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                raw_response=response,
                latency_ms=(perf_counter() - started) * 1000,
                stop_reason=getattr(response, "stop_reason", None),
                truncated=getattr(response, "stop_reason", None) == "max_tokens",
                requested_settings={key: value for key, value in {"temperature": self.config.temperature, "max_tokens": self.config.max_tokens, **provider_extras}.items() if value is not None},
effective_settings={key: value for key, value in {"temperature": self.config.temperature, "max_tokens": self.config.max_tokens, **provider_extras}.items() if value is not None},
                usage=usage_fields(getattr(response, "usage", None), MESSAGE_USAGE_FIELDS),
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model_name,
                provider="anthropic",
                error=str(e)
            )
        
    def is_available(self) -> bool:
        try:
            # Minimum test call
            self._client.messages.create(
                model=self.config.model_name,
                max_tokens=10,
                messages=[{"role":"user", "content":"Hi"}],
            )
            return True
        except Exception:
            return False

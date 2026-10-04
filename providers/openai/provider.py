"""
providers/openai/provider.py
OpenAI API provider
"""
from typing import Optional
from core.registry import ProviderRegistry
from core.base import BaseProvider, LLMResponse, usage_fields
from time import perf_counter

RESPONSES_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_tokens_details",
    "output_tokens_details",
)
CHAT_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
)


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts)
    return "" if content is None else str(content)

@ProviderRegistry.register("openai")
class OpenAIProvider(BaseProvider):

    def connect(self) -> bool:
        try:
            import openai
            self._client = openai.OpenAI(
                api_key = self.config.api_key,
                base_url = self.config.base_url,
                max_retries=0,
                timeout=self.config.extra_params.get("_provider_timeout_s", 180),
            )
            return True
        except ImportError:
            raise RuntimeError("openai package is not installed. Run: pip install openai")

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        messages=[]
        if system_prompt:
            messages.append({"role" : "system", "content" : system_prompt})
        messages.append({"role" : "user", "content" : prompt})

        settings = dict(self.config.extra_params)
        endpoint = settings.pop("_endpoint", "chat")
        settings.pop("_provider_timeout_s", None)
        requested = dict(settings)
        if self.config.temperature is not None:
            requested["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            requested["max_tokens"] = self.config.max_tokens
        if endpoint == "responses":
            response_settings = dict(settings)
            if self.config.temperature is not None:
                response_settings["temperature"] = self.config.temperature
            if self.config.max_tokens is not None:
                response_settings["max_output_tokens"] = self.config.max_tokens
            try:
                started = perf_counter()
                response = self._client.responses.create(
                    model=self.config.model_name, input=messages, **response_settings
                )
                usage = getattr(response, "usage", None)
                return LLMResponse(
                    content=response.output_text or "", model=self.config.model_name, provider="openai",
                    prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                    latency_ms=(perf_counter() - started) * 1000,
                    raw_response=response,
                    stop_reason=getattr(response, "status", None),
                    truncated=getattr(response, "status", None) == "incomplete",
                    requested_settings=requested,
                    effective_settings=response_settings,
                    usage=usage_fields(usage, RESPONSES_USAGE_FIELDS),
                )
            except Exception as exc:
                return LLMResponse("", self.config.model_name, "openai", error=str(exc), requested_settings=requested)
        chat_settings = dict(settings)
        if self.config.temperature is not None:
            chat_settings["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            chat_settings["max_completion_tokens"] = self.config.max_tokens
        try:
            started = perf_counter()
            response = self._client.chat.completions.create(
                model = self.config.model_name,
                messages = messages,
                **chat_settings,
            )
            return LLMResponse(
                content=_text_content(response.choices[0].message.content),
                model = self.config.model_name,
                provider = "openai",
                prompt_tokens = response.usage.prompt_tokens,
                completion_tokens = response.usage.completion_tokens,
                raw_response = response,
                latency_ms=(perf_counter() - started) * 1000,
                stop_reason=getattr(response.choices[0], "finish_reason", None),
                truncated=getattr(response.choices[0], "finish_reason", None) == "length",
                requested_settings=requested,
                effective_settings=chat_settings,
                usage=usage_fields(getattr(response, "usage", None), CHAT_USAGE_FIELDS),
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model_name,
                provider="openai",
                error=str(e),
                requested_settings=requested,
            )

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

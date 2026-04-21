"""
providers/openai/provider.py
OpenAI API provider
"""
from typing import Optional
from core.registry import ProviderRegistry
from core.base import BaseProvider, LLMResponse

@ProviderRegistry.register("openai")
class OpenAIProvider(BaseProvider):

    def connect(self) -> bool:
        try:
            import openai
            self._client = openai.OpenAI(
                api_key = self.config.api_key,
                base_url = self.config.base_url
            )
            return True
        except ImportError:
            raise RuntimeError("openai package is not installed. Run: pip install openai")

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        messages=[]
        if system_prompt:
            messages.append({"role" : "system", "content" : system_prompt})
        messages.append({"role" : "user", "content" : prompt})

        try:
            response = self._client.chat.completions.create(
                model = self.config.model_name,
                messages = messages,
                **self.config.extra_params,
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model = self.config.model_name,
                provider = "openai",
                prompt_tokens = response.usage.prompt_tokens,
                completion_tokens = response.usage.completion_tokens,
                raw_response = response,
            )
        except Exception as e:
            error_str = str(e)
            # Retry using /v1/responses for models that require it (e.g. gpt-5-codex)
            if "v1/responses" in error_str or "supported in v1/responses" in error_str:
                try:
                    input_text = "\n".join(m["content"] for m in messages)
                    response = self._client.responses.create(
                        model=self.config.model_name,
                        input=input_text,
                        **self.config.extra_params,
                    )
                    return LLMResponse(
                        content=response.output_text,
                        model=self.config.model_name,
                        provider="openai",
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                        raw_response=response,
                    )
                except Exception as retry_e:
                    return LLMResponse(
                        content="",
                        model=self.config.model_name,
                        provider="openai",
                        error=str(retry_e),
                    )
            # Retry using legacy /v1/completions for non-chat models (e.g. codex)
            if "not a chat model" in error_str or "v1/completions" in error_str:
                try:
                    prompt_text = "\n".join(m["content"] for m in messages)
                    response = self._client.completions.create(
                        model=self.config.model_name,
                        prompt=prompt_text,
                        **self.config.extra_params,
                    )
                    return LLMResponse(
                        content=response.choices[0].text,
                        model=self.config.model_name,
                        provider="openai",
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        raw_response=response,
                    )
                except Exception as retry_e:
                    return LLMResponse(
                        content="",
                        model=self.config.model_name,
                        provider="openai",
                        error=str(retry_e),
                    )
            return LLMResponse(
                content="",
                model=self.config.model_name,
                provider="openai",
                error = error_str
            )

    def is_available(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

"""
providers/antropic/provider.py
Anthropic API provider
"""
from typing import Optional
from core.registry import ProviderRegistry
from core.base import BaseProvider, LLMResponse
import asyncio

@ProviderRegistry.register("anthropic")
class AnthropicProvider(BaseProvider):

    def connect(self) -> bool:
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key = self.config.api_key
            )
            return True
        except ImportError:
            raise RuntimeError("anthropic package is not installed. Run: pip install anthropic")
        
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        kwargs = dict(
            model = self.config.model_name,
            max_tokens = self.config.max_tokens,
            messages = [{"role":"user", "content":prompt}],
            temperature = self.config.temperature,
            **self.config.extra_params,
        )    
        if system_prompt:
            kwargs["system"] = system_prompt
        try:
            response = self._client.messages.create(**kwargs)
            return LLMResponse(
                content=response.content[0].text,
                model=self.config.model_name,
                provider="anthropic",
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                raw_response=response,
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
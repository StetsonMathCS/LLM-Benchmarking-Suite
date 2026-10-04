"""
providers/ollama/provider.py
Ollama API provider
"""

from typing import Optional
from core.registry import ProviderRegistry
from core.base import BaseProvider, LLMResponse, usage_fields

@ProviderRegistry.register("ollama")
class OllamaProvider(BaseProvider):

    DEFAULT_BASE_URL = "http://localhost:11434"

    def connect(self) -> bool:
        try:
            import ollama
            self._base_url = self.config.base_url or self.DEFAULT_BASE_URL
            self._lib = ollama.Client(
                host=self._base_url,
                timeout=self.config.extra_params.get("_provider_timeout_s", 180),
                max_retries=0,
            )
            return True
        except ImportError:
            raise RuntimeError("ollama package is not installed. Run: pip install ollama")
        
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        messages=[]
        if system_prompt:
            messages.append({"role" : "system", "content" : system_prompt})
        messages.append({"role" : "user", "content" : prompt})

        try:
            options = dict(self.config.extra_params)
            options.pop("_provider_timeout_s", None)
            if self.config.temperature is not None:
                options["temperature"] = self.config.temperature
            if self.config.max_tokens is not None:
                options["num_predict"] = self.config.max_tokens
            response = self._lib.chat(
                model = self.config.model_name,
                messages = messages,
                options=options,
            )
            return LLMResponse(
                content=response["message"]["content"],
                model=self.config.model_name,
                provider="ollama",
                raw_response=response,
                stop_reason=response.get("done_reason"),
                truncated=response.get("done_reason") == "length",
                prompt_tokens=int(response.get("prompt_eval_count", 0) or 0),
                completion_tokens=int(response.get("eval_count", 0) or 0),
                latency_ms=float(response.get("total_duration", 0) or 0) / 1_000_000,
requested_settings={key: value for key, value in {"temperature": self.config.temperature, "max_tokens": self.config.max_tokens, **{k: v for k, v in self.config.extra_params.items() if not k.startswith("_")}}.items() if value is not None},
                effective_settings=options,
                usage=usage_fields(response, ("prompt_eval_count", "eval_count", "total_token_count")),
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.model_name,
                provider="ollama",
                error=str(e)
            )
        
    def is_available(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self._base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
    
    def list_local_models(self) -> list[str]:
        """Return list of models pulled in Ollama."""
        try:
            result = self._lib.list()
            return [m.model for m in result.models]
        except Exception as e:
            print("Error occured:",e)
            return []

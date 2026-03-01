"""
providers/ollama/provider.py
Ollama API provider
"""

from typing import Optional
from core.registry import ProviderRegistry
from core.base import BaseProvider, LLMResponse

@ProviderRegistry.register("ollama")
class OllamaProvider(BaseProvider):

    DEFAULT_BASE_URL = "http://localhost:11434"

    def connect(self) -> bool:
        try:
            import ollama
            self._lib = ollama
            self._base_url = self.config.base_url or self.DEFAULT_BASE_URL
            return True
        except ImportError:
            raise RuntimeError("ollama package is not installed. Run: pip install ollama")
        
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        messages=[]
        if system_prompt:
            messages.append({"role" : "system", "content" : system_prompt})
        messages.append({"role" : "user", "content" : prompt})

        try:
            response = self._lib.chat(
                model = self.config.model_name,
                messages = messages,
                options = {
                    "temperature" : self.config.temperature,
                    "num_predict" : self.config.max_tokens,
                    **self.config.extra_params,
                }
            )
            return LLMResponse(
                content=response["message"]["content"],
                model=self.config.model_name,
                provider="ollama"
                raw_response=response
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
            return [m["name"] for m in result.get("models", [])]
        except Exception:
            return []

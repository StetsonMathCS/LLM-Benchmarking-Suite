"""
providers/huggingface/provider.py
HuggingFace local model provider using transformeres pipeline.
Supports any causal LM loadable via AutoModelForCausalLM.
"""

from typing import Optional
from core.base import BaseProvider, ModelConfig, LLMResponse
from core.registry import ProviderRegistry

@ProviderRegistry.register("ollama")
class HuggingFaceProvider(BaseProvider):
    """
    Load any HuggingFace model locally.
    config.model_name can be:
      - A Hub ID:    "microsoft/phi-2"
      - A local path: "/path/to/model"
    Optional extra_params:
      - device_map: "auto" | "cpu" | "cuda" | "mps"
      - torch_dtype: "float16" | "bfloat16" | "float32"
      - load_in_8bit: True
      - load_in_4bit: True
    """
    
    def connect(self) -> bool:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except ImportError:
            raise RuntimeError("transformers/torch not installed. See requirements.txt")

        device_map = self.config.extra_params.get("device_map", "auto")
        dtype_str = self.config.extra_params.get("torch_dtype", "float32")
        dtype = getattr(torch, dtype_str, torch.float32)

        load_kwargs = dict(
            pretrained_model_name_or_path=self.config.model_name,
            device_map=device_map,
            torch_dtype=dtype,
            token=self.config.api_key,  # HF token if needed
        )
        if self.config.extra_params.get("load_in_8bit"):
            load_kwargs["load_in_8bit"] = True
        if self.config.extra_params.get("load_in_4bit"):
            load_kwargs["load_in_4bit"] = True

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, token=self.config.api_key
        )
        self._model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
        )
        return True

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        try:
            outputs = self._pipeline(
                full_prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                return_full_text=False,
            )
            content = outputs[0]["generated_text"]
            return LLMResponse(
                content=content,
                model=self.config.model_name,
                provider="huggingface",
            )
        except Exception as e:
            return LLMResponse(
                content="", model=self.config.model_name,
                provider="huggingface", error=str(e),
            )

    def is_available(self) -> bool:
        return self._model is not None and self._pipeline is not None

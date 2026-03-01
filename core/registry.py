"""
core/registry.py
Provider factory - instantiates the correct provider from a ModelConfig
"""

from core.base import BaseProvider, ModelConfig

class ProviderRegistry:
    """Maps provider strings to their implementation classes."""

    _registry: dict = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider class."""
        def decorator(provider_cls):
            cls._registry[name] = provider_cls
            return provider_cls
        return decorator
    
    @classmethod
    def create(cls, config: ModelConfig) -> BaseProvider:
        provider_cls = cls._registry.get(config.provider)
        if not provider_cls:
            raise ValueError(
                f"Unknown provider '{config.provider}'."
                f"Available: {list(cls._registry.keys())}"
            )
        instance = provider_cls(config)
        instance.connect()
        return instance
    
    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())
    
# Auto-Register all providers on import
def _register_all():
    # TO DO

_register_all()
"""
core/base.py
Abstract base classes for the entire testing suite.
All providers, benchmarks, and analyzers inherit from these.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
from time import perf_counter

class BenchmarkStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"

@dataclass 
class LLMResponse:
    """Standardized response from any LLM Provider"""
    content: str
    model: str
    provider: str
    prompt_tokens: int=0
    completion_tokens: int=0
    latency_ms: float=0.0
    raw_response: Any=None
    error: Optional[str]=None

    @property 
    def success(self) -> bool:
        return self.error is None and bool(self.content)

@dataclass
class ModelConfig:
    """Config for any LLM Provider."""
    provider: str # "openai" | "anthropic" | "HuggingFace" | "Ollama"
    model_name: str
    api_key: Optional[str]=None
    base_url: Optional[str]=None
    temperature: float=0.7
    max_tokens: int=4096
    extra_params: dict = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """The result of a single benchmark test"""
    benchmark_name: str
    status: BenchmarkStatus
    score: Optional[float]=None # 0.0-1.0 where applicable
    details: dict = field(default_factory=dict)
    issues_found: list = field(default_factory=list)
    raw_outputs: list = field(default_factory=list)
    duration_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
            return {
                "benchmark": self.benchmark_name,
                "status": self.status.value,
                "score": self.score,
                "issues_found": len(self.issues_found),
                "issues": self.issues_found,
                "details": self.details,
                "duration_s": self.duration_s,
            }


class BaseProvider(ABC):
    """Abstract base for all LLM Providers"""

    def __init__(self, config:ModelConfig):
        self.config = config
        self._client = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Initialize connection and return connection status (True/False)."""
        ...
    
    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str]=None) -> LLMResponse:
        """Send a prompt and return a standardized LLM response."""
        ...
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if model/provider is available at the moment."""
        ...
    
    def complete_timed(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Wrapper that records latency."""
        start = time.perf_counter()
        response = self.complete(prompt, system_prompt)
        response.latency_ms = (time.perf_counter() - start) * 1000
        return response

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name})"

class BaseBenchmark(ABC):
    """Abstract base for all benchmark tests."""

    name = "base_benchmark"
    desc = ""
    category = "general"

    def __init__(self, provider: BaseProvider):
        self.provider = provider
        self._results: list[BenchmarkResult] = []

    @abstractmethod
    def run(self, code_input: str, **kwargs) -> BenchmarkResult:
        """
        Execute the benchmark against the given code sample.
        Returns a BenchmarkResult.
        """
        ...

    @abstractmethod
    def generate_report(self) -> dict:
        """Summarize all results from this benchmark into a report dict."""
        ...

    def _timed_run(self, code_input: str, **kwargs) -> BenchmarkResult:
        start = time.perf_counter()
        result = self.run(code_input, **kwargs)
        result.duration_s = time.perf_counter() - start
        self._results.append(result)
        return result

    def get_history(self) -> list[BenchmarkResult]:
        return self._results.copy()
    
class BaseAnalyzer(ABC):
    """Abstract base for static analysis tools used inside benchmarks."""

    @abstractmethod
    def analyze(self, code: str, language: str = "python") -> list[dict]:
        """
        Analyze code and return a list of issue dicts with at least:
        { "severity": str, "message": str, "line": int, "rule": str }
        """
        ...
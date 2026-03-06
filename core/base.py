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
from pathlib import Path
from utils import code_runner
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
    """Config for any LLM Provider and task execution."""
    provider: str # "openai" | "anthropic" | "HuggingFace" | "Ollama"
    model_name: str
    api_key: Optional[str]=None
    base_url: Optional[str]=None
    temperature: float=0.7
    max_tokens: int=4096
    system_prompt: Optional[str]=None  # User-defined system instruction
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name})"

class BaseBenchmark(ABC):
    """
    Abstract base for all benchmarks.

    Each benchmark defines:
      - build_prompt()  → HOW to construct the user prompt for this task
      - run()           → full benchmark flow (prompt → LLM → evaluate)
      - generate_report()

    The system_prompt is supplied by the user via ModelConfig and
    wired automatically by execute().

    Flow:
        benchmark.execute(code_input)
          └─► provider.complete(
                  prompt       = benchmark.build_prompt(code_input),
                  system_prompt = self.model_config.system_prompt,
              )
    """

    name = "base_benchmark"
    desc = ""
    category = "general"

    def __init__(self, code_language:str, provider: BaseProvider, model_config: "ModelConfig"):
        self.provider = provider
        self.model_config = model_config
        self._results: list[BenchmarkResult] = []
        self.language = code_language

    def _load_prompt_template(self, language: str, filename_map: dict) -> Optional[str]:
        """Try to load a prompt template from prompts/ directory."""
        prompts_dir = Path(__file__).parent.parent / "prompts"
        path = prompts_dir / filename_map.get(language, "python")
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    @abstractmethod
    def build_prompt(self, language, code_input: str, **kwargs) -> str:
        """
        Build the user-facing prompt that contains the code sample
        and any task-specific context (language, constraints, …).
        """
        ...

    @abstractmethod
    def run(self, code_input: str, system_prompt:Optional[str], **kwargs) -> BenchmarkResult:
        """
        Execute the benchmark against the given code sample.
        Returns a BenchmarkResult.
        """
        ...

    def _timed_run(self, code_input: str, **kwargs) -> BenchmarkResult:
        start = perf_counter()
        prompt = self.build_prompt(self.language, code_input, **kwargs)
        system_prompt=self.model_config.system_prompt
        result = self.run(prompt, system_prompt, **kwargs)
        result.duration_s = perf_counter() - start
        self._results.append(result)
        return result

    def get_history(self) -> list[BenchmarkResult]:
        return self._results.copy()

    def generate_report(self) -> dict:
        return {
            "benchmark": self.name,
            "runs": len(self._results),
            "results": [r.to_dict() for r in self._results],
        }

@dataclass
class DimensionResult:
    """Score produced by a single evaluation dimension."""
    dimension_name: str
    score: float              # 0.0 – 1.0
    passed: bool = True       # used for dimension tests passing without error
    details: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)


class BaseDimension(ABC):
    """
    Defines HOW to measure one quality aspect of LLM output.

    A dimension receives the original code and the LLM-generated
    output and returns a DimensionResult.
    """
    runner = code_runner.CodeRunner
    name: str = "base_dimension"
    description: str = ""
    lang_map = {
        "python" : runner.run_python,
        "cpp" : runner.run_cpp,
        "javascript" : runner.run_javascript
    }
    @abstractmethod
    def evaluate(
        self,
        language:str,
        original_code: str,
        generated_code: str,
        **kwargs,
    ) -> DimensionResult:
        """Score the generated output on this quality dimension."""
        ...


class BaseAnalyzer(ABC):
    """Abstract base for static analysis tools used inside dimensions."""

    @abstractmethod
    def analyze(self, code: str, language: str = "python") -> list[dict]:
        """
        Analyze code and return a list of issue dicts with at least:
        { "severity": str, "message": str, "line": int, "rule": str }
        """
        ...
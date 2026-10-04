"""
core/base.py
Abstract base classes for the entire testing suite.
All providers, benchmarks, and analyzers inherit from these.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Iterable, Optional
from enum import Enum
from time import perf_counter
from pathlib import Path
import math
from utils import code_runner

def _read_field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _scalarize(value: Any) -> Any:
    """Reduce an SDK usage value to plain JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _scalarize(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    for method in ("model_dump", "to_dict", "dict"):
        dump = getattr(value, method, None)
        if callable(dump):
            try:
                return _scalarize(dump())
            except Exception:
                continue
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return {
            str(key): _scalarize(item)
            for key, item in data.items()
            if not str(key).startswith("_")
        }
    return str(value)


def usage_fields(source: Any, names: Iterable[str]) -> dict:
    """Copy provider-reported usage fields, flattening nested detail objects.

    Providers report the authoritative counts only; this preserves exactly what
    the API returned so accounting never has to estimate or re-derive values.
    """
    if source is None:
        return {}
    collected: dict = {}
    for name in names:
        value = _read_field(source, name)
        if value is None:
            continue
        collected[name] = _scalarize(value)
    return collected


DIMENSION_ID_BY_NAME = {
    "Code Consistency": "code_consistency",
    "Functional Correctness": "functional_correctness",
    "Linting": "linting",
    "Runtime Analysis": "runtime_analysis",
    "Vulnerabilities": "vulnerabilities",
    "Structural Similarity (SS)": "structural_similarity",
    "Reference Review Similarity (RRS)": "reference_review_similarity",
    "Reference Test Success (RTS)": "reference_test_success",
    "Generated Test Effectiveness (GTE)": "generated_test_effectiveness",
    # Accepted only at configuration/import boundaries. Migration retains the
    # historical label and evaluator provenance rather than relabeling scores.
    "Semantic Drift": "structural_similarity",
    "Code Review Quality": "reference_review_similarity",
    "Code Generation Tests": "reference_test_success",
    "Code Completion Tests": "reference_test_success",
    "Test Pass Rate": "generated_test_effectiveness",
}


def to_jsonable(value: Any) -> Any:
    """Convert FACETS result values to genuine JSON-compatible objects.

    Provider SDK response objects are deliberately not stringified: their
    useful, stable fields belong on ``LLMResponse`` and opaque SDK objects are
    represented by their type.  This prevents accidental secret/header dumps.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return {"opaque_type": f"{type(value).__module__}.{type(value).__name__}"}
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
    status: str = "completed"
    stop_reason: Optional[str] = None
    attempts: int = 1
    truncated: bool = False
    requested_settings: dict = field(default_factory=dict)
    effective_settings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    @property 
    def success(self) -> bool:
        return self.error is None and bool(self.content and self.content.strip())

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "attempts": self.attempts,
            "truncated": self.truncated,
            "requested_settings": to_jsonable(self.requested_settings),
            "effective_settings": to_jsonable(self.effective_settings),
            "usage": to_jsonable(self.usage),
        }

@dataclass
class ModelConfig:
    """Config for any LLM Provider and task execution."""
    provider: str # "openai" | "anthropic" | "HuggingFace" | "Ollama"
    model_name: str
    api_key: Optional[str]=None
    base_url: Optional[str]=None
    temperature: Optional[float]=None
    max_tokens: Optional[int]=None
    system_prompt: Optional[str]=None  # User-defined system instruction
    extra_params: dict = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """The result of a single benchmark test"""
    benchmark_name: str
    status: BenchmarkStatus
    combined_score: Optional[float]=None # 0.0-1.0 where applicable
    details: dict = field(default_factory=dict)
    issues_found: dict = field(default_factory=dict)
    duration_s: float = 0.0
    metadata: dict = field(default_factory=dict)
    llm_response: Optional[LLMResponse] = None

    def to_dict(self) -> dict:
            data = {
                "benchmark": self.benchmark_name,
                "status": self.status.value,
                "combined_score": self.combined_score,
                "issues_found": len(self.issues_found),
                "issues": self.issues_found,
                "details": self.details,
                "duration_s": self.duration_s,
                "metadata": self.metadata,
                "llm_response" : self.llm_response.to_dict() if self.llm_response else None,
            }
            return to_jsonable(data)


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
    code_input = ""
    system_prompt = ""

    def __init__(self, code_language:str, provider: BaseProvider):
        self.provider = provider
        self._results: list[BenchmarkResult] = []
        self.language = code_language
        self.system_prompt = self.provider.config.system_prompt

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
        self.code_input = code_input
        start = perf_counter()
        prompt = self.build_prompt(self.language, code_input, **kwargs)
        system_prompt=self.system_prompt
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
    dimension_id: str = ""
    status: str = "ok"  # ok | candidate_failure | infrastructure_error | not_applicable
    applicable: bool = True

    def __post_init__(self) -> None:
        if not self.dimension_id:
            self.dimension_id = DIMENSION_ID_BY_NAME.get(self.dimension_name, self.dimension_name)
        if not math.isfinite(float(self.score)) or not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(f"dimension score must be finite and in [0, 1], got {self.score!r}")

    def to_dict(self) -> dict:
        return to_jsonable({
            "dimension_id": self.dimension_id,
            "display_name": self.dimension_name,
            "score": self.score,
            "passed": self.passed,
            "status": self.status,
            "applicable": self.applicable,
            "details": self.details,
            "issues": self.issues,
        })


class BaseDimension(ABC):
    """
    Defines HOW to measure one quality aspect of LLM output.

    A dimension receives the original code and the LLM-generated
    output and returns a DimensionResult.
    """
    runner = code_runner.CodeRunner
    name: str = "base_dimension"
    dimension_id: str = "base_dimension"
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
        generated_code: str,
        original_code: Optional[str] = None,
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

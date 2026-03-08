"""
core/suite.py
Orchestrates benchmark runs - wires providers, benchmarks, and reporters together.
"""
from dataclasses import dataclass, field
from typing import Optional, Callable
import time
from benchmarks.tasks.bug_fixing import BugFixingBenchmark
from benchmarks.tasks.code_completion import CodeCompletionBenchmark
from benchmarks.tasks.code_generation import CodeGeneratinBenchmark  # Note: typo in class name
from benchmarks.tasks.code_review import CodeReviewBenchmark
from benchmarks.tasks.refactoring import RefactoringBenchmark
from benchmarks.tasks.test_generation import TestGenerationBenchmark
from benchmarks.tasks.translation import TranslationBenchmark
from benchmarks.tasks.partial_transform import PartialTransformBenchmark
from core.base import BaseProvider, BenchmarkResult, BaseBenchmark, BenchmarkStatus, ModelConfig

@dataclass
class SuiteConfig:
    """Top-level configuration for a test run."""
    name: str = "LLM Test Run"
    code_samples: list[str] = field(default_factory=list)
    selected_benchmarks: list[str] = field(default_factory=list) # empty = all
    language: str = "python" # python | javascript | C++
    output_dir: str = "./reports/output"
    parallel: bool = False
    progress_callback: Optional[Callable] = None

class TestSuite:
    """
    Central Orchestrator.
    Usage:
        suite = TestSuite(provider, config)
        suite.register_benchmark(SemanticDriftBenchmark(provider))
        results=suite.run_all(code_sample)
    """
    def __init__(self, provider: BaseProvider, config:SuiteConfig):
        self.provider = provider
        self.config = config
        self._benchmarks: dict[str, BaseBenchmark] = {}
        self._results: list[BenchmarkResult] = []
        self._start_time: Optional[float] = None

    def register_benchmarks(self) -> "TestSuite":
        """Auto-register every built-in benchmark."""
        self._benchmarks["bug_fixing"] =  BugFixingBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["code_completion"] = CodeCompletionBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["code_generation"] = CodeGeneratinBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["code_review"] = CodeReviewBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["refactoring"] = RefactoringBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["test_generation"] = TestGenerationBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["translation"] = TranslationBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["partial_transform"] = PartialTransformBenchmark(code_language=self.config.language, provider=self.provider)
        return self    
    
    def run_all(self) -> list[BenchmarkResult]:
        """Run all registered benchmarks against a single code sample"""
        self._start_time = time.perf_counter()
        self._results.clear()
        # TO DO Load relevant code datasets and run the benchmarks
        ...
    
    def get_summary(self) -> dict:
        total = len(self._results)
        passed = sum(1 for r in self._results if r.status == BenchmarkStatus.PASSED)
        failed = sum(1 for r in self._results if (r.status == BenchmarkStatus.ERROR or r.status == BenchmarkStatus.FAILED))
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0

        return {
            "suite_name": self.config.name,
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total else 0,
            "elapsed_s": elapsed,
            "results": [r.to_dict() for r in self._results],
        }

    @property
    def available_benchmarks(self) -> list[str]:
        return list(self._benchmarks.keys())
    
    
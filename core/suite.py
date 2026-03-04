"""
core/suite.py
Orchestrates benchmark runs - wires providers, benchmarks, and reporters together.
"""
from dataclasses import dataclass, field
from typing import Optional, Callable
import time

from core.base import BaseProvider, BenchmarkResult, BaseBenchmark, BenchmarkStatus, ModelConfig

@dataclass
class SuiteConfig:
    """Top-level configuration for a test run."""
    name: str = "LLM Test Run"
    code_samples: list[str] = field(default_factory=list)
    selected_benchmarks: list[str] = field(default_factory=list) # empty = all
    language: str = "python"
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

    # TO DO
    def register_benchmark(self, benchmark: BaseBenchmark) -> "TestSuite":
        """Auto-register every built-in benchmark."""
        # TO-DO once you make all the benchmarks
        return self
    
    def run_all(self, code_input: str) -> list[BenchmarkResult]:
        """Run all registered benchmarks against a single code sample"""
        self._start_time = time.perf_counter()
        self._results.clear()

        for name, benchmark in self._benchmarks.items():
            if self.config.progress_callback:
                self.config.progress_callback(name, "running")
            try:
                result = benchmark._timed_run(code_input, language = self.config.language)
            except Exception as e:
                result = BenchmarkResult(
                    benchmark_name=name,
                    status=BenchmarkStatus.ERROR,
                    details={"error": str(e)},
                )            
            self._results.append(result)
        if self.config.progress_callback:
            self.config.progress_callback(name, result.status.value)
        return self._results
    
    def run_benchmark(self, benchmark_name: str, code_input: str) -> Optional[BenchmarkResult]:
        """Run a single named benchmark."""
        bench = self._benchmarks.get(benchmark_name)
        if not bench:
            return None
        return bench._timed_run(code_input, language=self.config.language)
    
    def get_summary(self) -> dict:
        total = len(self._results)
        passed = sum(1 for r in self._results if r.status == BenchmarkStatus.PASSED)
        failed = sum(1 for r in self._results if r.status == BenchmarkStatus.FAILED)
        errored = sum(1 for r in self._results if r.status == BenchmarkStatus.ERROR)
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0

        return {
            "suite_name": self.config.name,
            "total": total,
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "pass_rate": passed / total if total else 0,
            "elapsed_s": elapsed,
            "results": [r.to_dict() for r in self._results],
        }

    @property
    def available_benchmarks(self) -> list[str]:
        return list(self._benchmarks.keys())
    
    
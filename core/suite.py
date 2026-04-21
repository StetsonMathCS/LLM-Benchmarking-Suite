"""
core/suite.py
Orchestrates benchmark runs - wires providers, benchmarks, and reporters together.
"""
from dataclasses import dataclass, field
from typing import Optional, Callable
import time
from benchmarks.tasks.bug_fixing import BugFixingBenchmark
from benchmarks.tasks.code_generation import CodeGenerationBenchmark
from benchmarks.tasks.code_review import CodeReviewBenchmark
from benchmarks.tasks.refactoring import RefactoringBenchmark
from benchmarks.tasks.test_generation import TestGenerationBenchmark
from benchmarks.tasks.translation import TranslationBenchmark
from core.base import BaseProvider, BenchmarkResult, BaseBenchmark, BenchmarkStatus, ModelConfig
from core.scoring import ScoringEngine
from datasets.mapper import DatasetMapper

@dataclass
class SuiteConfig:
    """Top-level configuration for a test run."""
    name: str = "LLM Test Run"
    selected_benchmarks: list[str] = field(default_factory=list)
    language: str = "python"
    output_dir: str = "./reports/outputs/python"
    parallel: bool = False
    progress_callback: Optional[Callable] = None
    target_language: Optional[str] = None
    # Scoring overrides — if None, ScoringEngine defaults are used
    task_weights: Optional[dict] = None
    pass_threshold: Optional[float] = None
    # pass@k configuration — num_samples > 1 enables multi-sample mode
    num_samples: int = 1
    pass_k_values: list[int] = field(default_factory=lambda: [1, 5, 10])

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
        self._mapper = DatasetMapper()

    def register_benchmarks(self) -> "TestSuite":
        """Auto-register every built-in benchmark."""
        self._benchmarks["bug_fixing"] =  BugFixingBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["code_generation"] = CodeGenerationBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["code_review"] = CodeReviewBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["refactoring"] = RefactoringBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["test_generation"] = TestGenerationBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["translation"] = TranslationBenchmark(code_language=self.config.language, provider=self.provider)
        return self
    
    def run_all(self) -> list[BenchmarkResult]:
        """Run all registered benchmarks against datasets from the mapper.

        When ``config.num_samples`` > 1, each record is sampled multiple times
        to enable pass@k estimation.  Every result is tagged with
        ``metadata["record_id"]`` and ``metadata["sample_index"]``.
        """
        self._start_time = time.perf_counter()
        self._results.clear()

        num_samples = max(1, self.config.num_samples)

        # Determine which benchmarks to run
        benchmarks_to_run = (
            self.config.selected_benchmarks
            if self.config.selected_benchmarks
            else list(self._benchmarks.keys())
        )

        # Execute each benchmark
        for task_name in benchmarks_to_run:
            if task_name not in self._benchmarks:
                print(f"Warning: Benchmark '{task_name}' not registered. Skipping.")
                continue

            try:
                # Load dataset for this task
                records = self._mapper.load_dataset(
                    task_name,
                    self.config.language
                )

                benchmark = self._benchmarks[task_name]

                # Run benchmark on each dataset record
                for record in records:
                    for sample_idx in range(num_samples):
                        try:
                            # Convert record to benchmark kwargs
                            kwargs = self._mapper.map_record_to_benchmark_kwargs(record)

                            # Add target_language for translation tasks
                            if task_name == "translation" and self.config.target_language:
                                kwargs["target_language"] = self.config.target_language

                            # Extract code_input and run benchmark
                            code_input = kwargs.pop("code_input", "")
                            result = benchmark._timed_run(
                                code_input=code_input,
                                **kwargs
                            )

                            # Tag so ScoringEngine can group correctly
                            result.metadata["task_name"] = task_name
                            result.metadata["record_id"] = record.record_id
                            result.metadata["sample_index"] = sample_idx

                            # Store result
                            self._results.append(result)

                            # Call progress callback if defined
                            if self.config.progress_callback:
                                display_id = (
                                    f"{record.record_id}[{sample_idx+1}/{num_samples}]"
                                    if num_samples > 1
                                    else record.record_id
                                )
                                self.config.progress_callback(
                                    task_name,
                                    display_id,
                                    result.status,
                                    result.combined_score,
                                )

                        except Exception as e:
                            # Log error and continue with next sample/record
                            error_result = BenchmarkResult(
                                benchmark_name=task_name,
                                status=BenchmarkStatus.ERROR,
                                details={"error": str(e), "record_id": record.record_id},
                                metadata={
                                    "task_name": task_name,
                                    "record_id": record.record_id,
                                    "sample_index": sample_idx,
                                },
                            )
                            self._results.append(error_result)
                            print(error_result)
                            print(f"Error in {task_name} record {record.record_id}: {e}")

            except FileNotFoundError:
                print(f"Dataset not found for task '{task_name}' and language '{self.config.language}'.")
            except ValueError as e:
                print(f"Error loading dataset for '{task_name}': {e}")
            except Exception as e:
                print(f"Unexpected error in task '{task_name}': {e}")

        return self._results
    
    def get_summary(self) -> dict:
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0

        report = ScoringEngine(
            self._results,
            task_weights=self.config.task_weights,
            pass_threshold=self.config.pass_threshold,
            num_samples=self.config.num_samples,
            pass_k_values=self.config.pass_k_values,
        ).compute()

        summary = {
            "suite_name": self.config.name,
            "elapsed_s": round(elapsed, 2),
            # Core scoring outputs
            "final_score": report.final_score,
            "grade": report.grade,
            "overall_pass_rate": report.overall_pass_rate,
            "category_scores": report.category_scores,
            "task_scores": {k: v.to_dict() for k, v in report.task_scores.items()},
            # Raw counts (kept for backward-compat with TUI)
            "total": report.total_records,
            "passed": report.total_passed,
            "failed": report.total_records - report.total_passed - report.total_errors,
            "errors": report.total_errors,
            "pass_rate": report.overall_pass_rate,  # alias
            "results": [r.to_dict() for r in self._results],
        }

        # Include pass@k when in multi-sample mode
        if report.pass_at_k:
            summary["pass_at_k"] = report.pass_at_k

        return summary

    @property
    def available_benchmarks(self) -> list[str]:
        return list(self._benchmarks.keys())
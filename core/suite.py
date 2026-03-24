"""
core/suite.py
Orchestrates benchmark runs - wires providers, benchmarks, and reporters together.
"""
from dataclasses import dataclass, field
from typing import Optional, Callable
import time
from benchmarks.tasks.bug_fixing import BugFixingBenchmark
from benchmarks.tasks.code_completion import CodeCompletionBenchmark
from benchmarks.tasks.code_generation import CodeGenerationBenchmark
from benchmarks.tasks.code_review import CodeReviewBenchmark
from benchmarks.tasks.refactoring import RefactoringBenchmark
from benchmarks.tasks.test_generation import TestGenerationBenchmark
from benchmarks.tasks.translation import TranslationBenchmark
from benchmarks.tasks.partial_transform import PartialTransformBenchmark
from core.base import BaseProvider, BenchmarkResult, BaseBenchmark, BenchmarkStatus, ModelConfig
from datasets.mapper import DatasetMapper

@dataclass
class SuiteConfig:
    """Top-level configuration for a test run."""
    name: str = "LLM Test Run"
    selected_benchmarks: list[str] = field(default_factory=list) # empty = all
    language: str = "python" # python | javascript | C++
    output_dir: str = f"./reports/outputs/{language}"
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
        self._mapper = DatasetMapper()

    def register_benchmarks(self) -> "TestSuite":
        """Auto-register every built-in benchmark."""
        self._benchmarks["bug_fixing"] =  BugFixingBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["code_completion"] = CodeCompletionBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["code_generation"] = CodeGenerationBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["code_review"] = CodeReviewBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["refactoring"] = RefactoringBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["test_generation"] = TestGenerationBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["translation"] = TranslationBenchmark(code_language=self.config.language, provider=self.provider)
        self._benchmarks["partial_transform"] = PartialTransformBenchmark(code_language=self.config.language, provider=self.provider)
        return self    
    
    def run_all(self) -> list[BenchmarkResult]:
        """Run all registered benchmarks against datasets from the mapper."""
        self._start_time = time.perf_counter()
        self._results.clear()
        
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
                    try:
                        # Convert record to benchmark kwargs
                        kwargs = self._mapper.map_record_to_benchmark_kwargs(record)
                        
                        # Extract code_input and run benchmark
                        code_input = kwargs.pop("code_input", "")
                        result = benchmark._timed_run(
                            code_input=code_input,
                            **kwargs
                        )
                        
                        # Store result
                        self._results.append(result)
                        
                        # Call progress callback if defined
                        if self.config.progress_callback:
                            self.config.progress_callback(
                                task_name, 
                                record.record_id, 
                                result.status
                            )
                    
                    except Exception as e:
                        # Log error and continue with next record
                        error_result = BenchmarkResult(
                            benchmark_name=task_name,
                            status=BenchmarkStatus.ERROR,
                            details={"error": str(e), "record_id": record.record_id}
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
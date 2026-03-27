"""
benchmarks/tasks/test_generation.py

Benchmarks that tests the Language Model on test generation tasks.
"""
from pathlib import Path
from typing import Optional
from core.base import (
    BaseProvider,
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkStatus,
    LLMResponse,
)
from benchmarks import matrix
from benchmarks.dimensions.test_pass_rate import TestPassRateDimension
class TestGenerationBenchmark(BaseBenchmark):
    name = "Test Generation Benchmark"
    desc = "Evaluate the Language Model on test generation tasks."
    category = "generation"

    def build_prompt(self, language: str, code_input: str, **kwargs) -> str:
        filename_map = {
            # Only python tests gen benchmarking available rn.
            "python": "python_test_gen.txt",
            # "cpp": "cpp_test_gen.txt",
            # "javascript": "javascript_test_gen.txt"
        }
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template.replace("{{CODE}}", code_input)
        return (
            f"Generate unit tests for the following {language} code:\n\n"
            f"```{language}\n{code_input}\n```"
        )

    def run(self, code_input: str, system_prompt: Optional[str], **kwargs) -> BenchmarkResult:
        llm_response = self.provider.complete(code_input, system_prompt)

        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error},
            )

        generated_tests = llm_response.content

        weights = matrix.DIMENSION_WEIGHTS["test_generation"] 
        dimensions = matrix.get_dimensions_for_task("test_generation")
        results = {}
        issues = {}
        combined_score = 0.00
        status = BenchmarkStatus.PASSED
        for cls in dimensions:
            dimension = cls()
            try:
                if isinstance(dimension, TestPassRateDimension):
                    result = dimension.evaluate(language=self.language, original_code=self.code_input, generated_tests=generated_tests, **kwargs)
                else:
                    result = dimension.evaluate(language=self.language, generated_code=generated_tests, **kwargs)
                print(result)
            except Exception as e:
                # Dimension evaluation failed - create error result
                from core.base import DimensionResult
                result = DimensionResult(
                    dimension_name=dimension.name,
                    score=0.0,
                    passed=False,
                    details={"error": str(e)},
                    issues=[str(e)]
                )

            if not result.passed :
                status = BenchmarkStatus.ERROR
                issues[dimension.name] = result.details.get("error", "Unknown error")
            results[dimension.name] = result
            # Calculating scores
            combined_score += (weights[dimension.name]*result.score) if result.score else 0.00
        return BenchmarkResult(
            benchmark_name=self.name,
            status=status,
            combined_score=combined_score,
            details=results,
            issues_found=issues,
            llm_response=llm_response,
        )

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
from core.scoring import PASS_THRESHOLD
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
            description = kwargs.get("description", "")
            desc_line = f"Context: {description}\n" if description else ""
            return template.replace("{{DESCRIPTION}}", desc_line).replace("{{CODE}}", code_input)
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
                details={"error": llm_response.error or "LLM returned empty response"},
                llm_response=llm_response,
            )

        from utils.code_runner import extract_code
        generated_tests = extract_code(llm_response.content)

        weights = matrix.DIMENSION_WEIGHTS["test_generation"]
        dimensions = matrix.get_dimensions_for_task("test_generation")
        results = {}
        issues = {}
        combined_score = 0.00
        for cls in dimensions:
            dimension = cls()
            try:
                if isinstance(dimension, TestPassRateDimension):
                    result = dimension.evaluate(language=self.language, original_code=self.code_input, generated_tests=generated_tests, **kwargs)
                else:
                    result = dimension.evaluate(language=self.language, generated_code=generated_tests, **kwargs)
            except Exception as e:
                from core.base import DimensionResult
                result = DimensionResult(
                    dimension_name=dimension.name,
                    score=0.0,
                    passed=False,
                    details={"error": str(e)},
                    issues=[str(e)]
                )
            if not result.passed:
                issues[dimension.name] = result.details.get("error") or f"Score below threshold ({result.score:.2f})"
            results[dimension.name] = result
            combined_score += (weights[dimension.name]*result.score) if result.score else 0.00
        status = BenchmarkStatus.PASSED if combined_score >= PASS_THRESHOLD else BenchmarkStatus.FAILED
        return BenchmarkResult(
            benchmark_name=self.name,
            status=status,
            combined_score=combined_score,
            details=results,
            issues_found=issues,
            llm_response=llm_response,
        )

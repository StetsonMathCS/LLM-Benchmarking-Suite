"""
benchmarks/tasks/bug_fixing.py

Task that asks the LLM to identify and fix bugs in a code snippet.
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
from benchmarks.dimensions.semantic_drift import SemanticDriftDimension
from benchmarks.dimensions.code_consistency import CodeConsistencyDimension
from core.scoring import PASS_THRESHOLD

class BugFixingBenchmark(BaseBenchmark):
    """Must provide expected output for the program as 'expected_output' arguement"""
    name = "Bug Fixing Benchmark"
    desc = "Evaluate the Language Model on bug fixing tasks."
    category = "transformation"

    def build_prompt(self, language: str, code_input: str, **kwargs) -> str:
        filename_map = {
            "python": "python_bugfix.txt",
            "cpp": "cpp_bugfix.txt",
            "javascript": "javascript_bugfix.txt"
        }
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template.replace("{{CODE}}", code_input)
        else:
            raise RuntimeError(f"Code bug fixing {language} template not available.")

    def run(self, prompt: str, system_prompt: Optional[str], **kwargs) -> BenchmarkResult:
        llm_response = self.provider.complete(prompt, system_prompt)

        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error or "LLM returned empty response"},
                llm_response=llm_response,
            )

        fixed_code = llm_response.content
        weights = matrix.DIMENSION_WEIGHTS["bug_fixing"]
        dimensions = matrix.get_dimensions_for_task("bug_fixing")
        results = {}
        issues = {}
        combined_score = 0.00
        for cls in dimensions:
            dimension = cls()
            try:
                if isinstance(dimension, SemanticDriftDimension):
                    result = dimension.evaluate(language=self.language, original_code=self.code_input, generated_code=fixed_code, **kwargs)
                else:
                    result = dimension.evaluate(language=self.language, generated_code=fixed_code, **kwargs)
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
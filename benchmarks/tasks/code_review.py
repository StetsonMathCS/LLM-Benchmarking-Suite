"""
benchmarks/tasks/code_review.py

Task that asks the LLM to review and provide feedback on code.
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
from core.scoring import PASS_THRESHOLD

class CodeReviewBenchmark(BaseBenchmark):
    name = "Code Review Benchmark"
    desc = "Evaluate the Language Model on code review tasks."
    category = "analysis"

    def build_prompt(self, language: str, code_input: str, **kwargs) -> str:
        filename_map = {
            "python": "python_review.txt",
            "cpp": "cpp_review.txt",
            "javascript": "javascript_review.txt"
        }
        review_requested = kwargs.get('review_request', '')
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template.replace("{{REVIEW_REQUEST}}", review_requested).replace("{{CODE}}", code_input)
        else:
            raise RuntimeError(f"Code review {language} template not available.")
    
    def run(self, prompt: str, system_prompt: Optional[str], **kwargs) -> BenchmarkResult:
        llm_response = self.provider.complete(prompt, system_prompt)
        
        if llm_response.error:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error or "LLM returned empty response"},
                llm_response=llm_response,
            )
        if not llm_response.content or not llm_response.content.strip():
            return BenchmarkResult(
                benchmark_name=self.name, status=BenchmarkStatus.FAILED, combined_score=0.0,
                details={"candidate_failure": "model returned empty output"}, llm_response=llm_response,
            )

        review = llm_response.content

        weights = matrix.DIMENSION_WEIGHTS["code_review"]
        dimensions = matrix.get_dimensions_for_task("code_review")
        results = {}
        issues = {}
        combined_score = 0.00
        infrastructure_error = False
        for cls in dimensions:
            dimension = cls()
            try:
                result = dimension.evaluate(generated_review=review, **kwargs)
            except Exception as e:
                from core.base import DimensionResult
                result = DimensionResult(
                    dimension_name=dimension.name,
                    score=0.0,
                    passed=False,
                    details={"error": str(e)},
                    issues=[str(e)],
                    status="infrastructure_error",
                )
            infrastructure_error = infrastructure_error or result.status == "infrastructure_error"
            if not result.passed:
                issues[dimension.dimension_id] = result.details.get("diagnostic") or result.details.get("error") or f"Score below threshold ({result.score:.2f})"
            results[dimension.dimension_id] = result
            combined_score += weights[dimension.dimension_id] * result.score
        status = BenchmarkStatus.ERROR if infrastructure_error else (BenchmarkStatus.PASSED if combined_score >= PASS_THRESHOLD else BenchmarkStatus.FAILED)
        return BenchmarkResult(
            benchmark_name=self.name,
            status=status,
            combined_score=None if infrastructure_error else combined_score,
            details=results,
            issues_found=issues,
            llm_response=llm_response,
        )

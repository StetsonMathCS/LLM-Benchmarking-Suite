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
        
        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error},
            )

        review = llm_response.content

        weights = matrix.DIMENSION_WEIGHTS["code_review"] 
        dimensions = matrix.get_dimensions_for_task("code_review")
        results = {}
        issues = {}
        combined_score = 0.00
        status = BenchmarkStatus.PASSED
        for cls in dimensions:
            dimension = cls()
            try:
                result = dimension.evaluate(generated_review=review, **kwargs)
                # print(result)
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
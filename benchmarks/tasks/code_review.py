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

        # TO DO evaluate on given dimensions via BenchmarkMatrix

        return BenchmarkResult(
            benchmark_name=self.name,
            status=BenchmarkStatus.PASSED,
            score=None,
            details={"review": review},
            raw_outputs=[llm_response],
        )

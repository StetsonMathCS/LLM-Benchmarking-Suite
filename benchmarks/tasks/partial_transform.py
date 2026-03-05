"""
benchmarks/tasks/partial_transform.py

Task that asks LLM to partially transform a code snippet
"""

from core.base import (
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkStatus
)

class PartialTransformBenchmark(BaseBenchmark):

    def build_prompt(self, language, code_input: str, **kwargs) -> str:
        filename_map = {
            "python": "transform.txt",
            "cpp": "transform.txt",
            "javascript" : "transform.txt"
        }
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template
        else:
            raise RuntimeError(f"Code generation {language} template not available.")
        
    def run(self, code_input: str, system_prompt: str | None, **kwargs) -> BenchmarkResult:
        llm_response = self.provider.complete(code_input, system_prompt)
        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error},
            )
        generated_code = llm_response.content

        # Evaluate with dimensions
        # TO DO: evaluate on given dimensions via BenchmarkMatrix

        # 3. Aggregate scores
        return BenchmarkResult(
            benchmark_name=self.name,
            status=BenchmarkStatus.PASSED,
            score=None,
            details={"transformed_code": generated_code},
            raw_outputs=[llm_response],
        )
    
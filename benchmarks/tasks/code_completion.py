"""
benchmarks/tasks/code_completion.py

Task that asks LM to complete a code snippet.
"""
from pathlib import Path
from typing import Optional
from core.base import (
    BaseProvider,
    BaseBenchmark, 
    BenchmarkResult,
    BenchmarkStatus,
    LLMResponse
)

class CodeCompletionBenchmark(BaseBenchmark):
    name = "Completion Benchmark"
    desc = "Evaluate the Language Model on code completion tasks."
    category = "generation"

    def build_prompt(self, language, code_input: str, **kwargs) -> str:
        filename_map = {
            "python" : "python_completion.txt",
            "cpp" : "cpp_completion.txt",
        }
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template
        else: 
            raise RuntimeError(f"Code completion {language} template not available.")
        
    def run(self, code_input: str, system_prompt: str | None, **kwargs) -> BenchmarkResult:
        llm_response = self.provider.complete(code_input, system_prompt)

        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error},
            )

        completed_code = llm_response.content

        # TO DO : run relevant dimesnions

        return BenchmarkResult(
            benchmark_name=self.name,
            status=BenchmarkStatus.PASSED,
            score=None,
            details={"completed_code":completed_code},
            raw_outputs=[llm_response]
        )

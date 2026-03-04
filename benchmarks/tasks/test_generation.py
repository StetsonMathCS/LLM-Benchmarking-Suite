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

class TestGenerationBenchmark(BaseBenchmark):
    name = "Test Generation Benchmark"
    desc = "Evaluate the Language Model on test generation tasks."
    category = "generation"

    def build_prompt(self, language: str, code_input: str, **kwargs) -> str:
        filename_map = {
            "python": "python_test_gen.txt",
            "cpp": "cpp_test_gen.txt"
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

        # TO DO evaluate on given dimensions via BenchmarkMatrix

        return BenchmarkResult(
            benchmark_name=self.name,
            status=BenchmarkStatus.PASSED,
            score=None,
            details={"generated_tests": generated_tests},
            raw_outputs=[llm_response],
        )


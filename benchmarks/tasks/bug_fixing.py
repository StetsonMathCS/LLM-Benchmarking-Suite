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

class BugFixingBenchmark(BaseBenchmark):
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
                details={"error": llm_response.error},
            )

        fixed_code = llm_response.content

        # TO DO evaluate on given dimensions via BenchmarkMatrix

        return BenchmarkResult(
            benchmark_name=self.name,
            status=BenchmarkStatus.PASSED,
            score=None,
            details={"fixed_code": fixed_code},
            raw_outputs=[llm_response],
        )

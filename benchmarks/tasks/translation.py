"""
benchmarks/tasks/translation.py

Task that asks the LLM to translate code between programming languages.
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

class TranslationBenchmark(BaseBenchmark):
    name = "Translation Benchmark"
    desc = "Evaluate the Language Model on code translation tasks."
    category = "transformation"

    def build_prompt(self, language: str, code_input: str, **kwargs) -> str:
        filename_map = {
            "python": "translate.txt",
            "cpp": "translate.txt",
            "javascript": "translate.txt"
        }
        target_language = kwargs.get("target_language", "python")
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template.replace("{{CODE}}", code_input).replace("{{TARGET}}", target_language)
        else:
            raise RuntimeError(f"Code translation {language} template not available.")

    def run(self, prompt: str, system_prompt: Optional[str], **kwargs) -> BenchmarkResult:
        llm_response = self.provider.complete(prompt, system_prompt)

        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error},
            )

        translated_code = llm_response.content

        # TO DO evaluate on given dimensions via BenchmarkMatrix

        return BenchmarkResult(
            benchmark_name=self.name,
            status=BenchmarkStatus.PASSED,
            score=None,
            details={"translated_code": translated_code},
            raw_outputs=[llm_response],
        )

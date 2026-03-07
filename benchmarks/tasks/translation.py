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
import matrix

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

        weights = matrix.DIMENSION_WEIGHTS["translation"] 
        dimensions = matrix.get_dimensions_for_task("translation")
        results = {}
        issues = {}
        combined_score = 0.00
        status = BenchmarkStatus.PASSED
        for cls in dimensions:
            dimension = cls()
            result = dimension.evaluate(language=self.language, generated_code=translated_code, **kwargs)
            if not result.passed :
                status = BenchmarkStatus.ERROR
                issues[dimension.name] = result.details["error"]
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

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
from benchmarks import matrix
from core.scoring import PASS_THRESHOLD

class TranslationBenchmark(BaseBenchmark):
    name = "Translation Benchmark"
    desc = "Evaluate the Language Model on Python to JavaScript to Python code translation tasks."
    category = "transformation"

    def build_prompt(self, language: str, code_input: str, **kwargs) -> str:
        if language != "python":
            raise RuntimeError(f"Translation task only supports Python source code. Received: {language}")
        
        target_language = kwargs.get("target_language", "javascript")
        if target_language not in ["javascript", "python"]:
            raise RuntimeError(f"Translation task only supports javascript and python as target languages. Received: {target_language}")
        
        filename_map = {
            "python": "translate.txt",
            "cpp": "translate.txt",
            "javascript": "translate.txt"
        }
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template.replace("{{CODE}}", code_input).replace("{{TARGET}}", target_language)
        else:
            raise RuntimeError(f"Code translation template not available.")

    def run(self, prompt: str, system_prompt: Optional[str], **kwargs) -> BenchmarkResult:
        llm_response = self.provider.complete(prompt, system_prompt)

        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error or "LLM returned empty response"},
                llm_response=llm_response,
            )

        from utils.code_runner import extract_code
        translated_code = extract_code(llm_response.content)
        
        # Get target language (what language the generated code is in)
        target_language = kwargs.get("target_language", "javascript")

        weights = matrix.DIMENSION_WEIGHTS["translation"]
        dimensions = matrix.get_dimensions_for_task("translation")
        results = {}
        issues = {}
        combined_score = 0.00
        for cls in dimensions:
            dimension = cls()
            try:
                result = dimension.evaluate(
                    language=self.language,
                    original_code=self.code_input,
                    generated_code=translated_code,
                    generated_code_language=target_language,
                    **kwargs
                )
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

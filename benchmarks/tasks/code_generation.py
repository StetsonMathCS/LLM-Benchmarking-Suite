"""
benchmarks/tasks/code_generation.py 

Task that asks the LLM to generate a code snippet.
"""
from pathlib import Path
from typing import Optional
from core.base import (
    BaseProvider,
    BaseBenchmark, 
    BenchmarkResult,
    BenchmarkStatus,
    LLMResponse,
    DimensionResult
)
from benchmarks import matrix
from benchmarks.dimensions.code_generation_tests import CodeGenerationTestsDimension
from core.scoring import PASS_THRESHOLD

class CodeGenerationBenchmark(BaseBenchmark):
    name = "Code Generation Benchmark"
    desc = "Evaluate the Language Model on code generation tasks."
    category = "generation"

    def build_prompt(self, language:str, code_input: str, **kwargs) -> str:
        filename_map = {
            "python": "generation.txt",
            "cpp": "generation.txt",
            "javascript": "generation.txt",
        }
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template.replace("{{LANG}}", language).replace("{{INPUT}}", code_input)
        else:
            raise RuntimeError(f"Code generation {language} template not available.")

    def run(self, prompt: str, system_prompt: str | None, **kwargs) -> BenchmarkResult:
        llm_response = self.provider.complete(prompt, system_prompt)
        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error or "LLM returned empty response"},
                llm_response=llm_response,
            )
        from utils.code_runner import extract_code
        generated_code = extract_code(llm_response.content)

        weights = matrix.DIMENSION_WEIGHTS["code_generation"]
        dimensions = matrix.get_dimensions_for_task("code_generation")
        results = {}
        issues = {}
        combined_score = 0.0
        active_weight = 0.0
        for cls in dimensions:
            dimension = cls()
            skipped = False
            try:
                if isinstance(dimension, CodeGenerationTestsDimension):
                    test = kwargs.get("test")
                    entry_point = kwargs.get("entry_point")
                    if test and entry_point and self.language == "python":
                        result = dimension.evaluate(
                            language=self.language,
                            generated_code=generated_code,
                            test=test,
                            entry_point=entry_point,
                        )
                    else:
                        # No tests available for this language/row — exclude from scoring
                        skipped = True
                        result = DimensionResult(
                            dimension_name=dimension.name,
                            score=0.0,
                            passed=False,
                            details={"skipped": "No test/entry_point provided or non-Python"}
                        )
                else:
                    # code_input is a natural-language prompt, not code — pass None
                    result = dimension.evaluate(
                        language=self.language,
                        original_code=None,
                        generated_code=generated_code,
                        **kwargs,
                    )
            except Exception as e:
                result = DimensionResult(
                    dimension_name=dimension.name,
                    score=0.0,
                    passed=False,
                    details={"error": str(e)},
                    issues=[str(e)]
                )
            if not skipped:
                active_weight += weights[dimension.name]
                combined_score += weights[dimension.name] * result.score
            if not result.passed and not skipped:
                issues[dimension.name] = result.details.get("error") or f"Score below threshold ({result.score:.2f})"
            results[dimension.name] = result

        # Renormalize so skipped dimensions don't dilute the score
        if active_weight > 0:
            combined_score /= active_weight
        status = BenchmarkStatus.PASSED if combined_score >= PASS_THRESHOLD else BenchmarkStatus.FAILED
        return BenchmarkResult(
            benchmark_name=self.name,
            status=status,
            combined_score=combined_score,
            details=results,
            issues_found=issues,
            llm_response=llm_response,
        )
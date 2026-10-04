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
from benchmarks.dimensions.reference_test_success import ReferenceTestSuccessDimension
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
        if llm_response.error:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error or "LLM returned empty response"},
                llm_response=llm_response,
            )
        if not llm_response.content or not llm_response.content.strip():
            return BenchmarkResult(
                benchmark_name=self.name, status=BenchmarkStatus.FAILED, combined_score=0.0,
                details={"candidate_failure": "model returned empty output"},
                llm_response=llm_response, metadata={"functional_correct": False},
            )
        from utils.code_runner import extract_code
        generated_code = extract_code(llm_response.content)

        weights = matrix.DIMENSION_WEIGHTS["code_generation"]
        dimensions = matrix.get_dimensions_for_task("code_generation")
        results = {}
        issues = {}
        combined_score = 0.0
        infrastructure_error = False
        for cls in dimensions:
            dimension = cls()
            try:
                if isinstance(dimension, ReferenceTestSuccessDimension):
                    result = dimension.evaluate(
                        language=self.language,
                        generated_code=generated_code,
                        test=kwargs.get("test"),
                        entry_point=kwargs.get("entry_point"),
                        dataset_hash=kwargs.get("dataset_hash"),
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
                    issues=[str(e)],
                    status="infrastructure_error",
                )
            infrastructure_error = infrastructure_error or result.status == "infrastructure_error"
            combined_score += weights[dimension.dimension_id] * result.score
            if not result.passed:
                issues[dimension.dimension_id] = result.details.get("diagnostic") or result.details.get("error") or f"Score below threshold ({result.score:.2f})"
            results[dimension.dimension_id] = result

        status = BenchmarkStatus.ERROR if infrastructure_error else (BenchmarkStatus.PASSED if combined_score >= PASS_THRESHOLD else BenchmarkStatus.FAILED)
        functional = results.get("reference_test_success")
        return BenchmarkResult(
            benchmark_name=self.name,
            status=status,
            combined_score=None if infrastructure_error else combined_score,
            details=results,
            issues_found=issues,
            llm_response=llm_response,
            metadata={"functional_correct": bool(functional and functional.score == 1.0 and functional.status == "ok")},
        )

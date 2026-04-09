"""
benchmarks/tasks/partial_transform.py

Task that asks LLM to partially transform a code snippet
"""

from core.base import (
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkStatus
)
from benchmarks import matrix
from benchmarks.dimensions import (
    partial_transform
)
from core.scoring import PASS_THRESHOLD
class PartialTransformBenchmark(BaseBenchmark):
    name = "Partial Transformation Benchmark"
    desc = "Evaluate the Language Model on code transformation tasks."
    category = "transformation"

    def build_prompt(self, language, code_input: str, **kwargs) -> str:
        filename_map = {
            "python": "transform.txt",
            "cpp": "transform.txt",
            "javascript" : "transform.txt"
        }
        transform_from = kwargs["transform_from"]
        transform_to = kwargs["transform_to"]
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template.replace("{{CODE}}",code_input).replace("{{FROM}}",transform_from).replace("{{TO}}",transform_to).replace("{{LANG}}", language)
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

        weights = matrix.DIMENSION_WEIGHTS["partial_transform"]
        dimensions = matrix.get_dimensions_for_task("partial_transform")
        results = {}
        issues = {}
        combined_score = 0.00
        for cls in dimensions:
            dimension = cls()
            try:
                result = dimension.evaluate(language=self.language, original_code=self.code_input, generated_code=generated_code, **kwargs)
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
    
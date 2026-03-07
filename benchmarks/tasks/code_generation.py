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
    LLMResponse
)
import matrix

class CodeGeneratinBenchmark(BaseBenchmark):
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
                details={"error": llm_response.error},
            )
        generated_code = llm_response.content

        weights = matrix.DIMENSION_WEIGHTS["code_completion"] 
        dimensions = matrix.get_dimensions_for_task("code_completion")
        results = {}
        issues = {}
        combined_score = 0.00
        status = BenchmarkStatus.PASSED
        for cls in dimensions:
            dimension = cls()
            result = dimension.evaluate(language=self.language, generated_code=generated_code, **kwargs)
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
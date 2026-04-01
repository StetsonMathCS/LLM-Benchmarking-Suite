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
from benchmarks.dimensions.code_completion_tests import CodeCompletionTestsDimension

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
                details={"error": llm_response.error},
            )
        generated_code = llm_response.content

        weights = matrix.DIMENSION_WEIGHTS["code_generation"] 
        dimensions = matrix.get_dimensions_for_task("code_generation")
        results = {}
        issues = {}
        combined_score = 0.00
        status = BenchmarkStatus.PASSED
        for cls in dimensions:
            dimension = cls()
            try:
                # CodeCompletionTestsDimension only runs for Python with test/entry_point
                if isinstance(dimension, CodeCompletionTestsDimension):
                    test = kwargs.get("test")
                    entry_point = kwargs.get("entry_point")
                    
                    # Only evaluate if test and entry_point are provided
                    if test and entry_point and self.language == "python":
                        # Pass test and entry_point directly, not in kwargs
                        eval_kwargs = {k: v for k, v in kwargs.items() if k not in ("test", "entry_point")}
                        result = dimension.evaluate(
                            language=self.language,
                            generated_code=generated_code,
                            test=test,
                            entry_point=entry_point,
                            **eval_kwargs
                        )
                    else:
                        # Skip this dimension if requirements not met
                        result = DimensionResult(
                            dimension_name=dimension.name,
                            score=1.0,
                            passed=True,
                            details={"skipped": "No test/entry_point provided or non-Python"}
                        )
                else:
                    result = dimension.evaluate(language=self.language, original_code=self.code_input, generated_code=generated_code, **kwargs)
                # print(result)
            except Exception as e:
                # Dimension evaluation failed - create error result
                result = DimensionResult(
                    dimension_name=dimension.name,
                    score=0.0,
                    passed=False,
                    details={"error": str(e)},
                    issues=[str(e)]
                )
            if not result.passed :
                status = BenchmarkStatus.ERROR
                issues[dimension.name] = result.details.get("error", "Unknown error")
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
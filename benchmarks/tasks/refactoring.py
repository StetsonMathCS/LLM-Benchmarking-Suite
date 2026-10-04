"""
benchmarks/tasks/refactoring.py

Task that asks the LLM to refactor a code snippet.
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
class RefactoringBenchmark(BaseBenchmark):
    name = "Refactoring Benchmark"
    desc = "Evaluate the Language Model on code refactoring tasks."
    category = "transformation"
    
    def build_prompt(self, language:str,  code_input: str, **kwargs) -> str:
        filename_map = {
            "python": "python_refactor.txt",
            "cpp": "cpp_refactor.txt",
            "javascript": "javascript_refactor.txt"
        }
        refactoring_task = kwargs.get("refactoring_task", "")
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template.replace("{{CODE}}", code_input).replace("{{TASK}}", refactoring_task)
        else:
            raise RuntimeError(f"Code refactoring {language} template not available.")


    def run(self, code_input: str, system_prompt:Optional[str] , **kwargs) -> BenchmarkResult:
        # 1. Execute the task
        
        llm_response = self.provider.complete(code_input, system_prompt)

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
        refactored_code = extract_code(llm_response.content)
        # 2. Evaluate with dimensions
        weights = matrix.DIMENSION_WEIGHTS["refactoring"]
        dimensions = matrix.get_dimensions_for_task("refactoring")
        results = {}
        issues = {}
        combined_score = 0.00
        infrastructure_error = False
        for cls in dimensions:
            dimension = cls()
            try:
                result = dimension.evaluate(language=self.language, original_code=self.code_input, generated_code=refactored_code, **kwargs)
            except Exception as e:
                from core.base import DimensionResult
                result = DimensionResult(
                    dimension_name=dimension.name,
                    score=0.0,
                    passed=False,
                    details={"error": str(e)},
                    issues=[str(e)],
                    status="infrastructure_error",
                )
            infrastructure_error = infrastructure_error or result.status == "infrastructure_error"
            if not result.passed:
                issues[dimension.dimension_id] = result.details.get("diagnostic") or result.details.get("error") or f"Score below threshold ({result.score:.2f})"
            results[dimension.dimension_id] = result
            combined_score += weights[dimension.dimension_id] * result.score
        status = BenchmarkStatus.ERROR if infrastructure_error else (BenchmarkStatus.PASSED if combined_score >= PASS_THRESHOLD else BenchmarkStatus.FAILED)
        functional = results.get("functional_correctness")
        return BenchmarkResult(
            benchmark_name=self.name,
            status=status,
            combined_score=None if infrastructure_error else combined_score,
            details=results,
            issues_found=issues,
            llm_response=llm_response,
            metadata={"functional_correct": bool(functional and functional.score == 1.0 and functional.status == "ok")},
        )

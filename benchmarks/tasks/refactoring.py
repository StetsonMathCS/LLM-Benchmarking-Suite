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
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template.replace("{{CODE}}", code_input)
        else:
            raise RuntimeError(f"Code refactoring {language} template not available.")


    def run(self, code_input: str, system_prompt:Optional[str] , **kwargs) -> BenchmarkResult:
        # 1. Execute the task
        
        llm_response = self.provider.complete(code_input, system_prompt)

        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error},
            )

        refactored_code = llm_response.content

        # 2. Evaluate with dimensions
        # TO DO: evaluate on given dimensions via BenchmarkMatrix

        # 3. Aggregate scores
        return BenchmarkResult(
            benchmark_name=self.name,
            status=BenchmarkStatus.PASSED,
            score=None,
            details={"refactored_code": refactored_code},
            raw_outputs=[llm_response],
        )

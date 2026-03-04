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
class CodeGeneratinBenchmark(BaseBenchmark):
    
    def build_prompt(self, language:str, code_input: str, **kwargs) -> str:
        filename_map = {
            "python": "generation.txt",
            "cpp": "generation.txt"
        }
        template = self._load_prompt_template(language, filename_map)
        if template:
            return template
        else:
            raise RuntimeError(f"Code generation {language} template not available.")

    def run(self, code_input: str, system_prompt: str | None, **kwargs) -> BenchmarkResult:
        llm_response = self.provider.complete(code_input, system_prompt)
        if not llm_response.success:
            return BenchmarkResult(
                benchmark_name=self.name,
                status=BenchmarkStatus.ERROR,
                details={"error": llm_response.error},
            )
        generated_code = llm_response.content

        # Evaluate with dimensions
        # TO DO: evaluate on given dimensions via BenchmarkMatrix

        # 3. Aggregate scores
        return BenchmarkResult(
            benchmark_name=self.name,
            status=BenchmarkStatus.PASSED,
            score=None,
            details={"generated_code": generated_code},
            raw_outputs=[llm_response],
        )
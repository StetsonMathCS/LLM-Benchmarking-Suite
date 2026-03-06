"""
benchmarks/tasks/partial_transform.py

Task that asks LLM to partially transform a code snippet
"""

from core.base import (
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkStatus
)

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
            details={"transformed_code": generated_code},
            raw_outputs=[llm_response],
        )
    
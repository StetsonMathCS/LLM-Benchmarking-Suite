"""
benchmarks/dimensions/functional_correctness.py

Dimension that evaluates functional correctness by comparing outputs.
"""
from core.base import (
    BaseDimension,
    DimensionResult
)
from utils.code_runner import CodeRunner
from typing import Optional
class FunctionalCorrectnessDimension(BaseDimension):
    name = "Functional Correctness"
    description = "Does generated code produce same output as original?"

    def evaluate(self, language: str, generated_code: str, original_code: Optional[str] = None, **kwargs) -> DimensionResult:
        """When you have only outputs and no original code (e.g. when you're checking for code generation) omit original_code and provide expected output in kwargs as original_output."""
        if not original_code or kwargs['expected_output']:
            expected_output = kwargs['expected_output']
            original_code=None
        try:
            if language == "python":
                orig_output = CodeRunner.run_python(original_code) if original_code else expected_output
                gen_output = CodeRunner.run_python(generated_code)
            elif language == "javascript":
                orig_output = CodeRunner.run_javascript(original_code) if original_code else expected_output
                gen_output = CodeRunner.run_javascript(generated_code)
            elif language == "cpp":
                orig_output = CodeRunner.run_cpp(original_code) if original_code else expected_output
                gen_output = CodeRunner.run_cpp(generated_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Unsupported language: {language}"}
                )

            score = 1.0 if orig_output.strip() == gen_output.strip() else 0.0

            return DimensionResult(
                dimension_name=self.name,
                score=score,
                details={
                    "original_output": orig_output,
                    "generated_output": gen_output,
                    "match": score == 1.0
                }
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )

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
        """Compare generated code output with expected output.
        
        For code transformations (partial_transform), compares against expected_output text.
        For other tasks, executes both original and generated code and compares outputs.
        """
        expected_output = kwargs.get('expected_output', None)
        
        # If we have expected_output and no original_code (or for transformations),
        # do text-based comparison
        if expected_output and not original_code:
            match = generated_code.strip() == expected_output.strip()
            score = 1.0 if match else 0.0
            return DimensionResult(
                dimension_name=self.name,
                score=score,
                details={
                    "original_output": expected_output,
                    "generated_output": generated_code,
                    "match": match
                }
            )
        
        # Otherwise, try to execute both versions
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
                    "original_output": orig_output if original_code else expected_output,
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

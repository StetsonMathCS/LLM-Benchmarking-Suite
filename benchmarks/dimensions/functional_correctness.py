"""
benchmarks/dimensions/functional_correctness.py

Dimension that evaluates functional correctness by comparing outputs.
"""
import re
from core.base import (
    BaseDimension,
    DimensionResult
)
from utils.code_runner import CodeRunner
from typing import Optional


def _normalize_output(text: str, language: str) -> str:
    """Normalize language-specific output differences for cross-language comparison."""
    if language == "javascript":
        # Boolean literals: JavaScript uses lowercase, Python uses capitalized
        text = re.sub(r'\btrue\b', 'True', text)
        text = re.sub(r'\bfalse\b', 'False', text)
        text = re.sub(r'\bnull\b', 'None', text)
        text = re.sub(r'\bundefined\b', 'None', text)
        # Collapse all whitespace (handles multiline arrays/objects from Node console.log)
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove spaces inside brackets and braces
        text = re.sub(r'\[\s+', '[', text)
        text = re.sub(r'\s+\]', ']', text)
        text = re.sub(r'\{\s+', '{', text)
        text = re.sub(r'\s+\}', '}', text)
        # Add single quotes around unquoted JS object keys: {cat: 3} -> {'cat': 3}
        text = re.sub(r'(\{|,\s*)([a-zA-Z_]\w*)\s*:', r"\1'\2':", text)
    return text


class FunctionalCorrectnessDimension(BaseDimension):
    name = "Functional Correctness"
    description = "Does generated code produce same output as original?"

    def evaluate(self, language: str, generated_code: str, original_code: Optional[str] = None, **kwargs) -> DimensionResult:
        """Compare generated code output with expected output.
        
        For code transformations (partial_transform), compares against expected_output text.
        For other tasks, executes both original and generated code and compares outputs.
        For translation tasks, uses generated_code_language for executing generated code.
        """
        expected_output = kwargs.get('expected_output', None)
        generated_code_language = kwargs.get('generated_code_language', language)
        
        # If we have expected_output and no original_code, execute the generated
        # code and compare its stdout to expected_output.
        if expected_output and not original_code:
            try:
                if generated_code_language == "python":
                    gen_output = CodeRunner.run_python(generated_code)
                elif generated_code_language == "javascript":
                    gen_output = CodeRunner.run_javascript(generated_code)
                elif generated_code_language == "cpp":
                    gen_output = CodeRunner.run_cpp(generated_code)
                else:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        details={"error": f"Unsupported language: {generated_code_language}"}
                    )
                match = _normalize_output(gen_output.strip(), generated_code_language) == _normalize_output(expected_output.strip(), language)
                score = 1.0 if match else 0.0
                return DimensionResult(
                    dimension_name=self.name,
                    score=score,
                    passed=match,
                    details={
                        "expected_output": expected_output,
                        "generated_output": gen_output,
                        "match": match
                    }
                )
            except Exception as e:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    passed=False,
                    details={"error": str(e)}
                )

        # Otherwise, try to execute both versions
        try:
            # Execute original code using source language
            if language == "python":
                orig_output = CodeRunner.run_python(original_code) if original_code else expected_output
            elif language == "javascript":
                orig_output = CodeRunner.run_javascript(original_code) if original_code else expected_output
            elif language == "cpp":
                orig_output = CodeRunner.run_cpp(original_code) if original_code else expected_output
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Unsupported source language: {language}"}
                )
            
            # Execute generated code using target language (may be different for translation tasks)
            if generated_code_language == "python":
                gen_output = CodeRunner.run_python(generated_code)
            elif generated_code_language == "javascript":
                gen_output = CodeRunner.run_javascript(generated_code)
            elif generated_code_language == "cpp":
                gen_output = CodeRunner.run_cpp(generated_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Unsupported generated code language: {generated_code_language}"}
                )

            norm_orig = _normalize_output(orig_output.strip(), language)
            norm_gen = _normalize_output(gen_output.strip(), generated_code_language)
            match = norm_orig == norm_gen
            score = 1.0 if match else 0.0

            return DimensionResult(
                dimension_name=self.name,
                score=score,
                passed=match,
                details={
                    "original_output": orig_output if original_code else expected_output,
                    "generated_output": gen_output,
                    "match": match
                }
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                passed=False,
                details={"error": str(e)}
            )

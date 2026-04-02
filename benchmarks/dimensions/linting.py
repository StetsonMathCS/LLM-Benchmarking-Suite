"""
benchmarks/dimensions/linting.py

Dimension that evaluates code quality via linting.
"""
import subprocess
import tempfile
from pathlib import Path
from core.base import (
    BaseDimension,
    DimensionResult
)
from typing import Optional

class LintingDimension(BaseDimension):
    name = "Linting"
    description = "Code quality violations: pylint (Python), cppcheck (C++), eslint (JS)."

    def evaluate(self, language: str, generated_code: str, original_code: Optional[str] = None, **kwargs) -> DimensionResult:
        try:
            # Get the language for the generated code (may differ from source language in translation tasks)
            generated_code_language = kwargs.get('generated_code_language', language)
            
            # If no original code provided, only lint generated code and return 1.0
            if not original_code:
                if generated_code_language == "python":
                    gen_violations = self._lint_python(generated_code)
                elif generated_code_language == "cpp":
                    gen_violations = self._lint_cpp(generated_code)
                elif generated_code_language == "javascript":
                    gen_violations = self._lint_javascript(generated_code)
                else:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        details={"error": f"Unsupported generated code language: {generated_code_language}"}
                    )
                
                return DimensionResult(
                    dimension_name=self.name,
                    score=1.0,
                    details={
                        "generated_violations": gen_violations,
                        "note": "No original code provided for comparison"
                    }
                )
            
            # Standard flow: compare original vs generated
            if language == "python":
                orig_violations = self._lint_python(original_code)
            elif language == "cpp":
                orig_violations = self._lint_cpp(original_code)
            elif language == "javascript":
                orig_violations = self._lint_javascript(original_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Unsupported source language: {language}"}
                )
            
            # Lint generated code using its own language
            if generated_code_language == "python":
                gen_violations = self._lint_python(generated_code)
            elif generated_code_language == "cpp":
                gen_violations = self._lint_cpp(generated_code)
            elif generated_code_language == "javascript":
                gen_violations = self._lint_javascript(generated_code)
            else:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Unsupported generated code language: {generated_code_language}"}
                )

            score = self._calculate_score(orig_violations, gen_violations)

            return DimensionResult(
                dimension_name=self.name,
                score=score,
                details={
                    "original_violations": orig_violations,
                    "generated_violations": gen_violations,
                    "regression": gen_violations > orig_violations
                }
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )

    @staticmethod
    def _lint_python(code: str) -> int:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                f.flush()
                filename = f.name

            result = subprocess.run(
                ["pylint", filename, "--disable=all", "--enable=W,E"],
                capture_output=True,
                text=True,
                timeout=10
            )
            Path(filename).unlink()

            output = result.stdout + result.stderr
            return output.count(':') if result.returncode != 0 else 0
        except FileNotFoundError:
            return 0
        except Exception:
            return 0

    @staticmethod
    def _lint_cpp(code: str) -> int:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False, encoding='utf-8') as f:
                f.write(code)
                f.flush()
                filename = f.name

            result = subprocess.run(
                ["cppcheck", "--enable=all", filename],
                capture_output=True,
                text=True,
                timeout=10
            )
            Path(filename).unlink()

            output = result.stdout + result.stderr
            lines = [l for l in output.split('\n') if 'error:' in l or 'warning:' in l]
            return len(lines)
        except FileNotFoundError:
            return 0
        except Exception:
            return 0

    @staticmethod
    def _lint_javascript(code: str) -> int:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8') as f:
                f.write(code)
                f.flush()
                filename = f.name

            result = subprocess.run(
                ["eslint", filename, "--format=json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            Path(filename).unlink()

            output = result.stdout
            if output:
                import json
                try:
                    data = json.loads(output)
                    violations = sum(len(f.get('messages', [])) for f in data)
                    return violations
                except json.JSONDecodeError:
                    return 0
            return 0
        except FileNotFoundError:
            return 0
        except Exception:
            return 0

    @staticmethod
    def _calculate_score(original_violations: int, generated_violations: int) -> float:
        if original_violations == 0 and generated_violations == 0:
            return 1.0
        if generated_violations > original_violations:
            return max(0.0, 1.0 - (generated_violations - original_violations) / max(original_violations, 1))
        return 1.0

"""
benchmarks/dimensions/code_generation_tests.py

Dimension that evaluates code generation by running tests against the generated code.
Uses entry_point and test fields from code generation datasets.
"""
import subprocess
import sys
import tempfile
from pathlib import Path
from core.base import (
    BaseDimension,
    DimensionResult
)
from typing import Optional


class CodeGenerationTestsDimension(BaseDimension):
    name = "Code Generation Tests"
    description = "Tests generated code against provided test cases using entry points."

    def evaluate(self, language: str, generated_code: str, **kwargs) -> DimensionResult:
        """
        Evaluate generated code by running it against test cases.

        Requires kwargs:
            - test: Test code as string
            - entry_point: Name of the function being tested
        """
        try:
            if language != "python":
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": "Code generation tests only supported for Python"}
                )

            test_code = kwargs.get("test")
            entry_point = kwargs.get("entry_point")

            if not test_code:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": "No test code provided"}
                )

            if not entry_point:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": "No entry point provided"}
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / "test_code.py"
                combined_code = generated_code + "\n\n" + test_code
                test_file.write_text(combined_code, encoding='utf-8')

                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )

                if result.returncode == 0:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=1.0,
                        passed=True,
                        details={
                            "tests_passed": True,
                            "output": result.stdout,
                            "entry_point": entry_point,
                        }
                    )
                else:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        passed=False,
                        details={
                            "tests_passed": False,
                            "error": result.stderr or result.stdout,
                            "entry_point": entry_point,
                        }
                    )

        except subprocess.TimeoutExpired:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                passed=False,
                details={"error": "Test execution timed out"}
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                passed=False,
                details={"error": str(e)}
            )

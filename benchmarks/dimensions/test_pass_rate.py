"""
benchmarks/dimensions/test_pass_rate.py

Dimension that evaluates test pass rate for generated tests.
"""
import subprocess
import tempfile
import os
from pathlib import Path
from core.base import (
    BaseDimension,
    DimensionResult
)

class TestPassRateDimension(BaseDimension):
    name = "Test Pass Rate"
    description = "Percentage of generated tests that pass against original code."

    def evaluate(self, language: str, original_code: str, generated_tests: str, **kwargs) -> DimensionResult:
        try:
            if language != "python":
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Test pass rate only supported for Python"}
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                orig_file = Path(tmpdir) / "code.py"
                test_file = Path(tmpdir) / "test_code.py"

                orig_file.write_text(original_code)
                test_file.write_text(generated_tests)

                result = subprocess.run(
                    ["pytest", str(test_file), "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )

                output = result.stdout + result.stderr
                passed, total = self._parse_pytest_output(output)

                if total == 0:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        details={"error": "No tests found", "output": output}
                    )

                score = passed / total

                return DimensionResult(
                    dimension_name=self.name,
                    score=score,
                    details={
                        "passed": passed,
                        "total": total,
                        "pass_rate": score
                    }
                )
        except subprocess.TimeoutExpired:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                details={"error": "Test execution timed out"}
            )
        except Exception as e:
            return DimensionResult(
                dimension_name=self.name,
                score=0.0,
                details={"error": str(e)}
            )

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple:
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line or 'failed' in line:
                if 'passed' in line and 'failed' in line:
                    parts = line.split()
                    passed = int(parts[0].split()[0])
                    failed = int([p for p in parts if 'failed' in p][0].split()[0])
                    return passed, passed + failed
                elif 'passed' in line:
                    parts = line.split()
                    passed = int(parts[0])
                    return passed, passed
        return 0, 0

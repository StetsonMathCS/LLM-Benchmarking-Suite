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
from typing import Optional 

class TestPassRateDimension(BaseDimension):
    name = "Test Pass Rate"
    description = "Percentage of generated tests that pass against original code."

    def evaluate(self, language: str, generated_tests: str, original_code:str, **kwargs) -> DimensionResult:
        try:
            if language != "python":
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": f"Test pass rate only supported for Python"}
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create combined test file with original code + generated tests
                combined_file = Path(tmpdir) / "test_combined.py"
                combined_code = f"{original_code}\n\n{generated_tests}"
                combined_file.write_text(combined_code)

                # Try pytest first
                result = subprocess.run(
                    ["python", "-m", "pytest", str(combined_file), "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )

                if result.returncode == 0 or "passed" in result.stdout.lower():
                    output = result.stdout + result.stderr
                    passed, total = self._parse_pytest_output(output)
                    
                    if total > 0:
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
                
                # Fallback: try unittest execution
                result = subprocess.run(
                    ["python", "-m", "unittest", "discover", "-s", tmpdir, "-p", "test_*.py", "-v"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=tmpdir
                )

                output = result.stdout + result.stderr
                passed, total = self._parse_unittest_output(output)

                if total == 0:
                    return DimensionResult(
                        dimension_name=self.name,
                        score=0.0,
                        details={"error": "No tests found or executed", "output": output}
                    )

                score = passed / total

                return DimensionResult(
                    dimension_name=self.name,
                    score=score,
                    details={
                        "passed": passed,
                        "total": total,
                        "pass_rate": score,
                        "method": "unittest"
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

    @staticmethod
    def _parse_unittest_output(output: str) -> tuple:
        """Parse unittest output to extract passed/total counts.
        
        Example output:
            test_add (test_combined.TestMath) ... ok
            test_fail (test_combined.TestMath) ... FAIL
            Ran 2 tests in 0.001s
            FAILED (failures=1)
        """
        lines = output.split('\n')
        total = 0
        passed = 0
        failed = 0
        
        # Count test results from individual test lines
        for line in lines:
            if '... ok' in line:
                passed += 1
            elif '... FAIL' in line or '... ERROR' in line:
                failed += 1
        
        # Get total from "Ran X tests" line
        for line in lines:
            if 'Ran' in line and 'tests' in line:
                try:
                    parts = line.split()
                    idx = parts.index('Ran')
                    total = int(parts[idx + 1])
                except:
                    pass
        
        # If we found individual results but no total, calculate it
        if total == 0 and (passed > 0 or failed > 0):
            total = passed + failed
        
        return max(0, passed), total

"""
benchmarks/dimensions/test_pass_rate.py

Dimension that evaluates test quality for generated tests using two signals:

1. Pass rate — what fraction of LLM-generated tests pass against the
   known-correct original code? Tests that fail on correct code are wrong.
2. Mutation kill rate — we create a null mutant (replace the function under
   test with one that returns None) and re-run the passing tests. Good tests
   should *fail* on the mutant. Tests that still pass are trivially true
   (e.g. no assertions, tautologies).

Final score = pass_rate * kill_rate. Both must be high for a good score.

When entry_point is not available, falls back to pass_rate only (original
behavior).
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from core.base import (
    BaseDimension,
    DimensionResult
)


class TestPassRateDimension(BaseDimension):
    name = "Test Pass Rate"
    description = "Percentage of generated tests that pass against original code, weighted by mutation detection."

    def evaluate(self, language: str, generated_tests: str, original_code: str, **kwargs) -> DimensionResult:
        try:
            if language != "python":
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": "Test pass rate only supported for Python"}
                )

            entry_point = kwargs.get("entry_point", "")

            # If generated_tests looks like HumanEval format (has check() function),
            # convert it to pytest format
            if "def check(" in generated_tests:
                generated_tests = self._convert_humaneval_to_pytest(generated_tests, entry_point)

            # ---- Step 1: run LLM tests against the real code ----
            passed, total, raw_output = self._run_tests(original_code, generated_tests)

            if total == 0:
                return DimensionResult(
                    dimension_name=self.name,
                    score=0.0,
                    details={"error": "No tests found or executed", "output": raw_output}
                )

            pass_rate = passed / total

            # If we don't have an entry_point, fall back to pass_rate only
            if not entry_point:
                return DimensionResult(
                    dimension_name=self.name,
                    score=pass_rate,
                    details={
                        "passed": passed,
                        "total": total,
                        "pass_rate": pass_rate,
                        "note": "No entry_point provided; mutation testing skipped"
                    }
                )

            # ---- Step 2: run LLM tests against a null mutant ----
            mutant_code = self._create_mutant(original_code, entry_point)
            mutant_passed, mutant_total, _ = self._run_tests(mutant_code, generated_tests)

            # Tests that passed on the original but fail on the mutant
            # detected the mutation — they are meaningful tests.
            if passed == 0:
                kill_rate = 0.0
            else:
                killed = passed - mutant_passed
                kill_rate = max(0.0, killed / passed)

            score = pass_rate * kill_rate

            return DimensionResult(
                dimension_name=self.name,
                score=score,
                details={
                    "passed_on_original": passed,
                    "total": total,
                    "pass_rate": pass_rate,
                    "passed_on_mutant": mutant_passed,
                    "killed": passed - mutant_passed,
                    "kill_rate": kill_rate,
                    "score": score,
                    "entry_point": entry_point,
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
    def _convert_humaneval_to_pytest(tests: str, entry_point: str) -> str:
        """Convert HumanEval check() format to pytest test_* format.

        HumanEval format:
            def check(candidate):
                assert candidate([1, 2]) == 3
                assert candidate([]) == 0

        Pytest format:
            def test_func_1():
                assert func([1, 2]) == 3
            def test_func_2():
                assert func([]) == 0
        """
        lines = tests.split("\n")
        asserts = [line for line in lines if line.strip().startswith("assert ")]

        if not asserts:
            return ""

        pytest_tests = []
        for i, assert_line in enumerate(asserts, 1):
            assert_line = assert_line.replace("candidate", entry_point)
            pytest_tests.append(f"def test_{entry_point}_{i}():")
            pytest_tests.append(f"    {assert_line}")

        return "\n".join(pytest_tests)

    @staticmethod
    def _create_mutant(original_code: str, entry_point: str) -> str:
        """Replace the target function with a null implementation.

        Appends a redefinition after the original code so that imports and
        helper definitions remain intact, but the function under test
        returns None for every input.
        """
        mutant_override = (
            f"\n# --- null mutant ---\n"
            f"def {entry_point}(*args, **kwargs):\n"
            f"    return None\n"
        )
        return original_code + mutant_override

    def _run_tests(self, code: str, tests: str, timeout: int = 30) -> tuple:
        """Run *tests* against *code* and return (passed, total, raw_output)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_combined.py"
            combined = f"{code}\n\n{tests}"
            test_file.write_text(combined, encoding="utf-8")

            # Try pytest first
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )

            output = result.stdout + result.stderr

            if result.returncode == 0 or "passed" in output.lower():
                passed, total = self._parse_pytest_output(output)
                if total > 0:
                    return passed, total, output

            # Fallback: unittest
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", tmpdir, "-p", "test_*.py", "-v"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            output = result.stdout + result.stderr
            passed, total = self._parse_unittest_output(output)
            return passed, total, output

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple:
        """Extract passed/total counts from pytest -v output.

        Pytest summary line looks like:
            '== 3 passed in 0.04s =='
            '== 2 failed, 5 passed in 0.06s =='
        """
        passed = 0
        failed = 0

        # Match the summary line at the end
        m = re.search(r"(\d+)\s+passed", output)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", output)
        if m:
            failed = int(m.group(1))

        total = passed + failed
        return passed, total

    @staticmethod
    def _parse_unittest_output(output: str) -> tuple:
        """Parse unittest -v output for passed/total counts.

        Individual lines:  test_foo ... ok | test_bar ... FAIL
        Summary:  Ran 5 tests in 0.003s
        """
        passed = 0
        failed = 0

        for line in output.split("\n"):
            if "... ok" in line:
                passed += 1
            elif "... FAIL" in line or "... ERROR" in line:
                failed += 1

        # Cross-check against "Ran N tests" line
        m = re.search(r"Ran\s+(\d+)\s+tests?", output)
        if m:
            total = int(m.group(1))
        else:
            total = passed + failed

        return passed, total

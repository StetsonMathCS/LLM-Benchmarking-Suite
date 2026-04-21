"""
End-to-end tests for the CodeGenerationTestsDimension.
"""
import pytest
from benchmarks.dimensions.code_generation_tests import CodeGenerationTestsDimension
from core.base import DimensionResult


@pytest.fixture
def dimension():
    return CodeGenerationTestsDimension()


# ── Full evaluation ───────────────────────────────────────────────────────

class TestCodeGenerationTests:
    def test_passing_tests_score_one(self, dimension):
        code = "def add(a, b):\n    return a + b\n"
        tests = "assert add(1, 2) == 3\nassert add(0, 0) == 0\n"
        result = dimension.evaluate(
            language="python",
            generated_code=code,
            test=tests,
            entry_point="add",
        )
        assert isinstance(result, DimensionResult)
        assert result.score == 1.0
        assert result.passed is True

    def test_failing_tests_score_zero(self, dimension):
        code = "def add(a, b):\n    return a - b\n"
        tests = "assert add(1, 2) == 3\n"
        result = dimension.evaluate(
            language="python",
            generated_code=code,
            test=tests,
            entry_point="add",
        )
        assert result.score == 0.0
        assert result.passed is False

    def test_no_test_code_returns_zero(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="def foo(): pass",
            entry_point="foo",
        )
        assert result.score == 0.0
        assert "No test code" in result.details.get("error", "")

    def test_no_entry_point_returns_zero(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="def foo(): pass",
            test="assert foo() is None",
        )
        assert result.score == 0.0
        assert "No entry point" in result.details.get("error", "")

    def test_non_python_returns_zero(self, dimension):
        result = dimension.evaluate(
            language="javascript",
            generated_code="function foo() {}",
            test="assert(foo());",
            entry_point="foo",
        )
        assert result.score == 0.0
        assert "error" in result.details

    def test_syntax_error_in_code(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="def add(a, b\n",
            test="assert add(1, 2) == 3\n",
            entry_point="add",
        )
        assert result.score == 0.0
        assert result.passed is False

    def test_timeout_handled(self, dimension):
        code = "import time\ndef slow():\n    time.sleep(100)\n"
        tests = "slow()\n"
        result = dimension.evaluate(
            language="python",
            generated_code=code,
            test=tests,
            entry_point="slow",
        )
        assert isinstance(result, DimensionResult)
        assert result.score == 0.0

    def test_details_contain_entry_point(self, dimension):
        code = "def greet():\n    return 'hi'\n"
        tests = "assert greet() == 'hi'\n"
        result = dimension.evaluate(
            language="python",
            generated_code=code,
            test=tests,
            entry_point="greet",
        )
        assert result.details.get("entry_point") == "greet"

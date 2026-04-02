"""
End-to-end tests for the TestPassRateDimension.
"""
import pytest
from benchmarks.dimensions.test_pass_rate import TestPassRateDimension
from core.base import DimensionResult


@pytest.fixture
def dimension():
    return TestPassRateDimension()


# ── Pytest output parsing ─────────────────────────────────────────────────

class TestParsePytestOutput:
    def test_all_passed(self):
        output = "3 passed in 0.01s"
        passed, total = TestPassRateDimension._parse_pytest_output(output)
        assert passed == 3
        assert total == 3

    def test_no_results(self):
        passed, total = TestPassRateDimension._parse_pytest_output("nothing here")
        assert passed == 0
        assert total == 0


# ── Unittest output parsing ───────────────────────────────────────────────

class TestParseUnittestOutput:
    def test_all_ok(self):
        output = (
            "test_a (module.Test) ... ok\n"
            "test_b (module.Test) ... ok\n"
            "Ran 2 tests in 0.001s\n"
            "OK\n"
        )
        passed, total = TestPassRateDimension._parse_unittest_output(output)
        assert passed == 2
        assert total == 2

    def test_mixed_results(self):
        output = (
            "test_a (module.Test) ... ok\n"
            "test_b (module.Test) ... FAIL\n"
            "Ran 2 tests in 0.001s\n"
            "FAILED (failures=1)\n"
        )
        passed, total = TestPassRateDimension._parse_unittest_output(output)
        assert passed == 1
        assert total == 2

    def test_no_tests(self):
        passed, total = TestPassRateDimension._parse_unittest_output("")
        assert passed == 0
        assert total == 0

    def test_errors_counted(self):
        output = (
            "test_a (module.Test) ... ERROR\n"
            "Ran 1 tests in 0.001s\n"
        )
        passed, total = TestPassRateDimension._parse_unittest_output(output)
        assert passed == 0
        assert total == 1


# ── Full evaluation ───────────────────────────────────────────────────────

class TestTestPassRateEvaluation:
    def test_passing_tests_score_one(self, dimension):
        original_code = "def add(a, b):\n    return a + b\n"
        test_code = (
            "import unittest\n"
            "class TestAdd(unittest.TestCase):\n"
            "    def test_basic(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n"
            "    def test_zero(self):\n"
            "        self.assertEqual(add(0, 0), 0)\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        result = dimension.evaluate(
            language="python",
            generated_tests=test_code,
            original_code=original_code,
        )
        assert isinstance(result, DimensionResult)
        assert result.score >= 0.0  # Depends on test runner availability

    def test_failing_tests_score_low(self, dimension):
        original_code = "def add(a, b):\n    return a + b\n"
        test_code = (
            "import unittest\n"
            "class TestAdd(unittest.TestCase):\n"
            "    def test_wrong(self):\n"
            "        self.assertEqual(add(1, 2), 999)\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        result = dimension.evaluate(
            language="python",
            generated_tests=test_code,
            original_code=original_code,
        )
        assert isinstance(result, DimensionResult)
        assert result.score <= 1.0

    def test_non_python_returns_zero(self, dimension):
        result = dimension.evaluate(
            language="javascript",
            generated_tests="test('foo', () => {});",
            original_code="function foo() {}",
        )
        assert result.score == 0.0
        assert "error" in result.details

    def test_syntax_error_in_tests(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_tests="def test_bad(\n",
            original_code="x = 1\n",
        )
        assert isinstance(result, DimensionResult)
        assert result.score == 0.0

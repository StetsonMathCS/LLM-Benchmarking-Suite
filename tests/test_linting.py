"""
End-to-end tests for the LintingDimension.
"""
import pytest
from benchmarks.dimensions.linting import LintingDimension
from core.base import DimensionResult


@pytest.fixture
def dimension():
    return LintingDimension()


# ── Score calculation ─────────────────────────────────────────────────────

class TestCalculateScore:
    def test_both_zero_returns_one(self):
        assert LintingDimension._calculate_score(0, 0) == 1.0

    def test_no_regression_returns_one(self):
        assert LintingDimension._calculate_score(5, 3) == 1.0

    def test_equal_violations_returns_one(self):
        assert LintingDimension._calculate_score(5, 5) == 1.0

    def test_regression_penalised(self):
        score = LintingDimension._calculate_score(2, 5)
        assert 0.0 <= score < 1.0

    def test_large_regression_approaches_zero(self):
        score = LintingDimension._calculate_score(1, 100)
        assert score <= 0.1

    def test_zero_original_many_generated(self):
        score = LintingDimension._calculate_score(0, 10)
        assert score < 1.0


# ── Python linting ────────────────────────────────────────────────────────

class TestPythonLinting:
    def test_clean_code_no_regression(self, dimension):
        clean = "x = 1\ny = 2\nprint(x + y)\n"
        result = dimension.evaluate(
            language="python",
            generated_code=clean,
            original_code=clean,
        )
        assert isinstance(result, DimensionResult)
        assert result.score >= 0.0

    def test_no_original_code_returns_one(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="x = 1\n",
        )
        assert result.score == 1.0
        assert "No original code" in result.details.get("note", "")

    def test_generated_code_language_override(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="const x = 1;",
            generated_code_language="javascript",
        )
        assert isinstance(result, DimensionResult)


# ── C++ linting ───────────────────────────────────────────────────────────

class TestCppLinting:
    def test_cpp_evaluates(self, dimension):
        code = "#include <iostream>\nint main() { return 0; }\n"
        result = dimension.evaluate(
            language="cpp",
            generated_code=code,
            original_code=code,
        )
        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0


# ── JavaScript linting ────────────────────────────────────────────────────

class TestJavaScriptLinting:
    def test_js_evaluates(self, dimension):
        code = 'const x = 1;\nconsole.log(x);\n'
        result = dimension.evaluate(
            language="javascript",
            generated_code=code,
            original_code=code,
        )
        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0


# ── Edge cases ────────────────────────────────────────────────────────────

class TestLintingEdgeCases:
    def test_unsupported_language(self, dimension):
        result = dimension.evaluate(
            language="rust",
            generated_code="fn main() {}",
            original_code="fn main() {}",
        )
        assert result.score == 0.0
        assert "error" in result.details

    def test_empty_code(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="",
            original_code="",
        )
        assert isinstance(result, DimensionResult)

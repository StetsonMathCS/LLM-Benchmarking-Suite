"""
End-to-end tests for the CodeConsistencyDimension.
"""
import pytest
from benchmarks.dimensions.code_consistency import CodeConsistencyDimension
from core.base import DimensionResult


@pytest.fixture
def dimension():
    return CodeConsistencyDimension()


# ── Python checks ──────────────────────────────────────────────────────────

class TestPythonConsistency:
    def test_consistent_snake_case_scores_high(self, dimension):
        code = """\
def calculate_sum(first_value, second_value):
    total_sum = first_value + second_value
    return total_sum
"""
        result = dimension.evaluate(language="python", generated_code=code)
        assert isinstance(result, DimensionResult)
        assert result.score >= 0.66

    def test_mixed_naming_scores_lower(self, dimension):
        code = """\
def calculateSum(first_value, secondValue):
    totalSum = first_value + secondValue
    return totalSum
"""
        result = dimension.evaluate(language="python", generated_code=code)
        assert isinstance(result, DimensionResult)
        assert result.score < 1.0

    def test_mixed_indentation_penalised(self, dimension):
        code = "def foo():\n    x = 1\n\ty = 2\n"
        result = dimension.evaluate(language="python", generated_code=code)
        assert result.score < 1.0

    def test_consistent_indentation_ok(self, dimension):
        code = "def foo():\n    x = 1\n    y = 2\n"
        result = dimension.evaluate(language="python", generated_code=code)
        assert result.score >= 0.66

    def test_empty_code_does_not_crash(self, dimension):
        result = dimension.evaluate(language="python", generated_code="")
        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0


# ── C++ checks ─────────────────────────────────────────────────────────────

class TestCppConsistency:
    def test_consistent_cpp_scores_high(self, dimension):
        code = """\
#include <iostream>
int main() {
    int my_var = 10;
    std::cout << my_var;
    return 0;
}
"""
        result = dimension.evaluate(language="cpp", generated_code=code)
        assert isinstance(result, DimensionResult)
        assert result.score >= 0.5

    def test_mixed_brace_style(self, dimension):
        code = """\
void foo() {
    int x = 1;
}
void bar()
{
    int y = 2;
}
"""
        result = dimension.evaluate(language="cpp", generated_code=code)
        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0

    def test_cpp_alias_accepted(self, dimension):
        code = "int main() { return 0; }"
        result = dimension.evaluate(language="c++", generated_code=code)
        assert isinstance(result, DimensionResult)


# ── JavaScript checks ─────────────────────────────────────────────────────

class TestJavaScriptConsistency:
    def test_consistent_js_scores_high(self, dimension):
        code = """\
function add(a, b) {
    return a + b;
}
const result = add(1, 2);
console.log(result);
"""
        result = dimension.evaluate(language="javascript", generated_code=code)
        assert isinstance(result, DimensionResult)
        assert result.score >= 0.5

    def test_missing_semicolons_penalised(self, dimension):
        code = """\
function add(a, b) {
    return a + b
}
const result = add(1, 2)
console.log(result)
"""
        result = dimension.evaluate(language="javascript", generated_code=code)
        assert isinstance(result, DimensionResult)
        assert result.score < 1.0

    def test_js_alias_accepted(self, dimension):
        code = "const x = 1;"
        result = dimension.evaluate(language="js", generated_code=code)
        assert isinstance(result, DimensionResult)


# ── Language edge cases ────────────────────────────────────────────────────

class TestLanguageEdgeCases:
    def test_unsupported_language_returns_zero(self, dimension):
        result = dimension.evaluate(language="rust", generated_code="fn main() {}")
        assert result.score == 0.0
        assert "error" in result.details

    def test_generated_code_language_override(self, dimension):
        """When generated_code_language kwarg is set, that language checker runs."""
        js_code = "const x = 1;"
        result = dimension.evaluate(
            language="python",
            generated_code=js_code,
            generated_code_language="javascript",
        )
        assert isinstance(result, DimensionResult)

    def test_original_code_ignored(self, dimension):
        """CodeConsistency only evaluates generated_code; original_code is unused."""
        result = dimension.evaluate(
            language="python",
            generated_code="x = 1\ny = 2\n",
            original_code="z = 3\n",
        )
        assert isinstance(result, DimensionResult)
        assert result.score >= 0.0

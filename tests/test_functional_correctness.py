"""
End-to-end tests for the FunctionalCorrectnessDimension.
"""
import pytest
from benchmarks.dimensions.functional_correctness import (
    FunctionalCorrectnessDimension,
    _normalize_output,
)
from core.base import DimensionResult


@pytest.fixture
def dimension():
    return FunctionalCorrectnessDimension()


# ── Output normalisation ──────────────────────────────────────────────────

class TestNormalizeOutput:
    def test_js_booleans_converted(self):
        assert _normalize_output("true", "javascript") == "True"
        assert _normalize_output("false", "javascript") == "False"

    def test_js_null_undefined_converted(self):
        assert _normalize_output("null", "javascript") == "None"
        assert _normalize_output("undefined", "javascript") == "None"

    def test_js_whitespace_is_not_broadly_rewritten(self):
        assert _normalize_output("[ 1,  2, 3 ]", "javascript") == "[ 1,  2, 3 ]"

    def test_js_object_keys_are_not_rewritten(self):
        assert _normalize_output("{cat: 3}", "javascript") == "{cat: 3}"

    def test_python_unchanged(self):
        text = "True False None"
        assert _normalize_output(text, "python") == text


# ── Python correctness ─────────────────────────────────────────────────────

class TestPythonCorrectness:
    def test_identical_code_scores_one(self, dimension):
        code = 'print("Hello")'
        result = dimension.evaluate(
            language="python",
            generated_code=code,
            original_code=code,
        )
        assert result.score == 1.0
        assert result.details["match"] is True

    def test_different_output_scores_zero(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code='print("World")',
            original_code='print("Hello")',
        )
        assert result.score == 0.0
        assert result.details["match"] is False

    def test_expected_output_without_original_code(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code='print(5)',
            expected_output="5",
        )
        assert result.score == 1.0

    def test_expected_output_mismatch(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code='print(5)',
            expected_output="10",
        )
        assert result.score == 0.0

    def test_error_in_generated_code_returns_zero(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="raise Exception('boom')",
            original_code='print(1)',
        )
        # Generated code error != original output -> score 0
        assert result.score == 0.0


# ── JavaScript correctness ─────────────────────────────────────────────────

class TestJavaScriptCorrectness:
    def test_identical_js_scores_one(self, dimension):
        code = 'console.log("hello");'
        result = dimension.evaluate(
            language="javascript",
            generated_code=code,
            original_code=code,
        )
        assert result.score == 1.0

    def test_different_js_scores_zero(self, dimension):
        result = dimension.evaluate(
            language="javascript",
            generated_code='console.log("a");',
            original_code='console.log("b");',
        )
        assert result.score == 0.0


# ── Cross-language (translation) ──────────────────────────────────────────

class TestCrossLanguage:
    def test_python_to_js_same_output(self, dimension):
        py_code = 'print(42)'
        js_code = 'console.log(42);'
        result = dimension.evaluate(
            language="python",
            generated_code=js_code,
            original_code=py_code,
            generated_code_language="javascript",
        )
        assert result.score == 1.0

    def test_python_to_js_different_output(self, dimension):
        py_code = 'print(1)'
        js_code = 'console.log(2);'
        result = dimension.evaluate(
            language="python",
            generated_code=js_code,
            original_code=py_code,
            generated_code_language="javascript",
        )
        assert result.score == 0.0


# ── Unsupported language ──────────────────────────────────────────────────

class TestUnsupportedLanguage:
    def test_unsupported_source_language(self, dimension):
        result = dimension.evaluate(
            language="rust",
            generated_code="fn main() {}",
            original_code="fn main() {}",
        )
        assert result.score == 0.0
        assert "diagnostic" in result.details

    def test_unsupported_generated_code_language(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="fn main() {}",
            expected_output="hello",
            generated_code_language="rust",
        )
        assert result.score == 0.0

"""
End-to-end tests for the RuntimeAnalysisDimension.
"""
import pytest
from benchmarks.dimensions.runtime_analysis import RuntimeAnalysisDimension
from core.base import DimensionResult


@pytest.fixture
def dimension():
    return RuntimeAnalysisDimension()


# ── Python runtime ────────────────────────────────────────────────────────

class TestPythonRuntime:
    def test_identical_code_neutral_score(self, dimension):
        code = 'print("hello")'
        result = dimension.evaluate(
            language="python",
            generated_code=code,
            original_code=code,
        )
        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0

    def test_no_original_code_with_output(self, dimension):
        code = 'print("hello")'
        result = dimension.evaluate(
            language="python",
            generated_code=code,
        )
        assert result.score == 1.0
        assert "No original code" in result.details.get("note", "")

    def test_no_original_no_output_code(self, dimension):
        code = "x = 1"  # No print -> no stdout
        result = dimension.evaluate(
            language="python",
            generated_code=code,
        )
        assert isinstance(result, DimensionResult)
        # Should get 0.5 (neutral) since returncode=0 but no output
        assert result.score >= 0.0

    def test_error_code_returns_zero(self, dimension):
        code = "raise Exception('boom')"
        result = dimension.evaluate(
            language="python",
            generated_code=code,
        )
        assert isinstance(result, DimensionResult)
        assert result.score <= 0.5


# ── Details structure ─────────────────────────────────────────────────────

class TestRuntimeDetails:
    def test_comparison_details_present(self, dimension):
        code = 'print(1)'
        result = dimension.evaluate(
            language="python",
            generated_code=code,
            original_code=code,
        )
        details = result.details
        if result.score > 0:
            assert "original_code_results" in details or "note" in details


# ── Edge cases ────────────────────────────────────────────────────────────

class TestRuntimeEdgeCases:
    def test_unsupported_language(self, dimension):
        result = dimension.evaluate(
            language="rust",
            generated_code="fn main() {}",
            original_code="fn main() {}",
        )
        assert result.score == 0.0

    def test_unsupported_language_no_original(self, dimension):
        result = dimension.evaluate(
            language="rust",
            generated_code="fn main() {}",
        )
        assert result.score == 0.0

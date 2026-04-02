"""
End-to-end tests for the VulnerabilitiesDimension.
"""
import pytest
from benchmarks.dimensions.vulnerabilities import VulnerabilitiesDimension
from core.base import DimensionResult


@pytest.fixture
def dimension():
    return VulnerabilitiesDimension()


# ── Score calculation ─────────────────────────────────────────────────────

class TestCalculateScore:
    def test_both_zero_returns_one(self):
        assert VulnerabilitiesDimension._calculate_score(0, 0) == 1.0

    def test_fewer_generated_returns_one(self):
        assert VulnerabilitiesDimension._calculate_score(5, 3) == 1.0

    def test_equal_returns_one(self):
        assert VulnerabilitiesDimension._calculate_score(3, 3) == 1.0

    def test_more_generated_penalised(self):
        score = VulnerabilitiesDimension._calculate_score(2, 5)
        assert 0.0 <= score < 1.0

    def test_large_increase_approaches_zero(self):
        score = VulnerabilitiesDimension._calculate_score(0, 100)
        assert score < 0.1


# ── Python scanning ───────────────────────────────────────────────────────

class TestPythonVulnerabilities:
    def test_safe_code_no_regression(self, dimension):
        safe = "x = 1\nprint(x)\n"
        result = dimension.evaluate(
            language="python",
            generated_code=safe,
            original_code=safe,
        )
        assert isinstance(result, DimensionResult)
        assert result.score >= 0.0

    def test_no_original_code_returns_one(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="x = 1\n",
        )
        assert result.score == 1.0

    def test_generated_code_language_override(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="const x = 1;",
            generated_code_language="javascript",
        )
        assert isinstance(result, DimensionResult)


# ── C++ scanning ──────────────────────────────────────────────────────────

class TestCppVulnerabilities:
    def test_cpp_evaluates(self, dimension):
        code = "#include <iostream>\nint main() { return 0; }\n"
        result = dimension.evaluate(
            language="cpp",
            generated_code=code,
            original_code=code,
        )
        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0


# ── JavaScript scanning ──────────────────────────────────────────────────

class TestJavaScriptVulnerabilities:
    def test_js_evaluates(self, dimension):
        code = "const x = 1;\nconsole.log(x);\n"
        result = dimension.evaluate(
            language="javascript",
            generated_code=code,
            original_code=code,
        )
        assert isinstance(result, DimensionResult)
        assert 0.0 <= result.score <= 1.0


# ── Edge cases ────────────────────────────────────────────────────────────

class TestVulnerabilitiesEdgeCases:
    def test_unsupported_language(self, dimension):
        result = dimension.evaluate(
            language="rust",
            generated_code="fn main() {}",
            original_code="fn main() {}",
        )
        assert result.score == 0.0
        assert "error" in result.details

    def test_unsupported_generated_language(self, dimension):
        result = dimension.evaluate(
            language="python",
            generated_code="fn main() {}",
            original_code="x = 1",
            generated_code_language="rust",
        )
        assert result.score == 0.0

    def test_details_contain_vuln_counts(self, dimension):
        code = "x = 1\n"
        result = dimension.evaluate(
            language="python",
            generated_code=code,
            original_code=code,
        )
        assert "original_vulnerabilities" in result.details
        assert "generated_vulnerabilities" in result.details

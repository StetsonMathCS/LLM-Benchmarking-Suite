"""
End-to-end tests for the TestGenerationBenchmark task.
"""
import pytest
from benchmarks.tasks.test_generation import TestGenerationBenchmark
from core.base import BenchmarkResult, BenchmarkStatus
from tests.conftest import FakeProvider


SOURCE_CODE = """\
def multiply(a, b):
    return a * b
"""

GENERATED_TESTS = """\
import unittest

class TestMultiply(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(multiply(2, 3), 6)

    def test_zero(self):
        self.assertEqual(multiply(0, 5), 0)

    def test_negative(self):
        self.assertEqual(multiply(-2, 3), -6)

if __name__ == "__main__":
    unittest.main()
"""


@pytest.fixture
def benchmark():
    provider = FakeProvider(response_content=GENERATED_TESTS)
    return TestGenerationBenchmark(code_language="python", provider=provider)


@pytest.fixture
def benchmark_error():
    provider = FakeProvider(error="server error")
    return TestGenerationBenchmark(code_language="python", provider=provider)


# ── Prompt building ───────────────────────────────────────────────────────

class TestTestGenerationPrompt:
    def test_build_prompt_python(self, benchmark):
        prompt = benchmark.build_prompt("python", SOURCE_CODE)
        assert "multiply" in prompt

    def test_build_prompt_with_description(self, benchmark):
        prompt = benchmark.build_prompt("python", SOURCE_CODE, description="multiplication function")
        assert "multiply" in prompt

    def test_build_prompt_fallback_for_unknown_language(self, benchmark):
        """Non-Python should use fallback prompt."""
        prompt = benchmark.build_prompt("go", SOURCE_CODE)
        assert "go" in prompt.lower()


# ── Full run ──────────────────────────────────────────────────────────────

class TestTestGenerationRun:
    def test_successful_run(self, benchmark):
        result = benchmark._timed_run(SOURCE_CODE)
        assert isinstance(result, BenchmarkResult)
        assert result.status in (BenchmarkStatus.PASSED, BenchmarkStatus.FAILED)
        assert result.combined_score is not None

    def test_error_response(self, benchmark_error):
        result = benchmark_error._timed_run(SOURCE_CODE)
        assert result.status == BenchmarkStatus.ERROR

    def test_dimensions_evaluated(self, benchmark):
        result = benchmark._timed_run(SOURCE_CODE)
        # Should have CodeConsistency, TestPassRate, Linting
        assert len(result.details) > 0

    def test_report_generation(self, benchmark):
        benchmark._timed_run(SOURCE_CODE)
        report = benchmark.generate_report()
        assert report["benchmark"] == "Test Generation Benchmark"
        assert report["runs"] == 1

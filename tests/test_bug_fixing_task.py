"""
End-to-end tests for the BugFixingBenchmark task.
"""
import pytest
from benchmarks.tasks.bug_fixing import BugFixingBenchmark
from core.base import BenchmarkResult, BenchmarkStatus, DimensionResult
from tests.conftest import FakeProvider


BUGGY_CODE = """\
def add(a, b):
    return a - b

print(add(2, 3))
"""

FIXED_CODE = """\
def add(a, b):
    return a + b

print(add(2, 3))
"""


@pytest.fixture
def benchmark_with_fix():
    provider = FakeProvider(response_content=FIXED_CODE)
    return BugFixingBenchmark(code_language="python", provider=provider)


@pytest.fixture
def benchmark_with_error():
    provider = FakeProvider(error="API rate limit exceeded")
    return BugFixingBenchmark(code_language="python", provider=provider)


# ── Prompt building ───────────────────────────────────────────────────────

class TestBugFixingPrompt:
    def test_build_prompt_python(self, benchmark_with_fix):
        prompt = benchmark_with_fix.build_prompt("python", BUGGY_CODE)
        assert BUGGY_CODE in prompt

    def test_build_prompt_cpp(self):
        provider = FakeProvider(response_content="int main() { return 0; }")
        bm = BugFixingBenchmark(code_language="cpp", provider=provider)
        prompt = bm.build_prompt("cpp", "#include <iostream>\nint main() {}")
        assert "#include" in prompt

    def test_build_prompt_javascript(self):
        provider = FakeProvider(response_content="function foo() {}")
        bm = BugFixingBenchmark(code_language="javascript", provider=provider)
        prompt = bm.build_prompt("javascript", "function foo() {}")
        assert "function" in prompt

    def test_build_prompt_unsupported_raises(self, benchmark_with_fix):
        with pytest.raises(RuntimeError):
            benchmark_with_fix.build_prompt("rust", "fn main() {}")


# ── Full run ──────────────────────────────────────────────────────────────

class TestBugFixingRun:
    def test_successful_run_returns_result(self, benchmark_with_fix):
        result = benchmark_with_fix._timed_run(
            BUGGY_CODE,
            expected_output="5",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.status in (BenchmarkStatus.PASSED, BenchmarkStatus.FAILED)
        assert result.combined_score is not None
        assert 0.0 <= result.combined_score <= 1.0
        assert result.duration_s > 0

    def test_error_response_returns_error_status(self, benchmark_with_error):
        result = benchmark_with_error._timed_run(
            BUGGY_CODE,
            expected_output="5",
        )
        assert result.status == BenchmarkStatus.ERROR
        assert "error" in result.details

    def test_llm_response_attached(self, benchmark_with_fix):
        result = benchmark_with_fix._timed_run(
            BUGGY_CODE,
            expected_output="5",
        )
        assert result.llm_response is not None
        assert result.llm_response.provider == "fake"

    def test_dimensions_evaluated(self, benchmark_with_fix):
        result = benchmark_with_fix._timed_run(
            BUGGY_CODE,
            expected_output="5",
        )
        # details should contain dimension results
        assert len(result.details) > 0

    def test_history_tracked(self, benchmark_with_fix):
        benchmark_with_fix._timed_run(BUGGY_CODE, expected_output="5")
        benchmark_with_fix._timed_run(BUGGY_CODE, expected_output="5")
        history = benchmark_with_fix.get_history()
        assert len(history) == 2

    def test_generate_report(self, benchmark_with_fix):
        benchmark_with_fix._timed_run(BUGGY_CODE, expected_output="5")
        report = benchmark_with_fix.generate_report()
        assert report["benchmark"] == "Bug Fixing Benchmark"
        assert report["runs"] == 1

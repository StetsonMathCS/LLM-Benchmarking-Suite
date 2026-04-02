"""
End-to-end tests for the CodeCompletionBenchmark task.
"""
import pytest
from benchmarks.tasks.code_completion import CodeCompletionBenchmark
from core.base import BenchmarkResult, BenchmarkStatus
from tests.conftest import FakeProvider


PARTIAL_CODE = """\
def fibonacci(n):
    if n <= 1:
        return n
"""

COMPLETED_CODE = """\
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

TEST_CODE = """\
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(5) == 5
"""


@pytest.fixture
def benchmark():
    provider = FakeProvider(response_content=COMPLETED_CODE)
    return CodeCompletionBenchmark(code_language="python", provider=provider)


@pytest.fixture
def benchmark_error():
    provider = FakeProvider(error="connection refused")
    return CodeCompletionBenchmark(code_language="python", provider=provider)


# ── Prompt building ───────────────────────────────────────────────────────

class TestCodeCompletionPrompt:
    def test_build_prompt_python(self, benchmark):
        prompt = benchmark.build_prompt("python", PARTIAL_CODE)
        assert "fibonacci" in prompt

    def test_build_prompt_cpp(self):
        provider = FakeProvider(response_content="int main() { return 0; }")
        bm = CodeCompletionBenchmark(code_language="cpp", provider=provider)
        prompt = bm.build_prompt("cpp", "int main() {")
        assert "int main" in prompt

    def test_build_prompt_javascript(self):
        provider = FakeProvider(response_content="function foo() {}")
        bm = CodeCompletionBenchmark(code_language="javascript", provider=provider)
        prompt = bm.build_prompt("javascript", "function foo() {")
        assert "function" in prompt


# ── Full run ──────────────────────────────────────────────────────────────

class TestCodeCompletionRun:
    def test_successful_run(self, benchmark):
        result = benchmark._timed_run(PARTIAL_CODE)
        assert isinstance(result, BenchmarkResult)
        assert result.status in (BenchmarkStatus.PASSED, BenchmarkStatus.FAILED)
        assert result.combined_score is not None
        assert 0.0 <= result.combined_score <= 1.0

    def test_run_with_tests(self, benchmark):
        result = benchmark._timed_run(
            PARTIAL_CODE,
            test=TEST_CODE,
            entry_point="fibonacci",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.combined_score is not None

    def test_error_response(self, benchmark_error):
        result = benchmark_error._timed_run(PARTIAL_CODE)
        assert result.status == BenchmarkStatus.ERROR

    def test_code_input_stored(self, benchmark):
        benchmark._timed_run(PARTIAL_CODE)
        assert benchmark.code_input == PARTIAL_CODE

    def test_llm_response_attached(self, benchmark):
        result = benchmark._timed_run(PARTIAL_CODE)
        assert result.llm_response is not None
        assert result.llm_response.content == COMPLETED_CODE

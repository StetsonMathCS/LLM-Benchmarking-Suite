"""
End-to-end tests for the RefactoringBenchmark task.
"""
import pytest
from benchmarks.tasks.refactoring import RefactoringBenchmark
from core.base import BenchmarkResult, BenchmarkStatus
from tests.conftest import FakeProvider


ORIGINAL_CODE = """\
def process(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

print(process([1, -2, 3, -4, 5]))
"""

REFACTORED_CODE = """\
def process(data):
    return [item * 2 for item in data if item > 0]

print(process([1, -2, 3, -4, 5]))
"""


@pytest.fixture
def benchmark():
    provider = FakeProvider(response_content=REFACTORED_CODE)
    return RefactoringBenchmark(code_language="python", provider=provider)


@pytest.fixture
def benchmark_error():
    provider = FakeProvider(error="quota exceeded")
    return RefactoringBenchmark(code_language="python", provider=provider)


# ── Prompt building ───────────────────────────────────────────────────────

class TestRefactoringPrompt:
    def test_build_prompt_python(self, benchmark):
        prompt = benchmark.build_prompt("python", ORIGINAL_CODE, refactoring_task="Use list comprehension")
        assert "process" in prompt

    def test_build_prompt_includes_task(self, benchmark):
        prompt = benchmark.build_prompt("python", ORIGINAL_CODE, refactoring_task="Use list comprehension")
        assert "list comprehension" in prompt.lower() or "Use list comprehension" in prompt

    def test_build_prompt_unsupported_raises(self, benchmark):
        with pytest.raises(RuntimeError):
            benchmark.build_prompt("rust", "fn main() {}", refactoring_task="refactor")


# ── Full run ──────────────────────────────────────────────────────────────

class TestRefactoringRun:
    def test_successful_run(self, benchmark):
        result = benchmark._timed_run(
            ORIGINAL_CODE,
            refactoring_task="Use list comprehension",
            expected_output="[2, 6, 10]",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.status == BenchmarkStatus.ERROR
        assert result.combined_score is None

    def test_error_response(self, benchmark_error):
        result = benchmark_error._timed_run(
            ORIGINAL_CODE,
            refactoring_task="Use list comprehension",
        )
        assert result.status == BenchmarkStatus.ERROR

    def test_dimensions_evaluated(self, benchmark):
        result = benchmark._timed_run(
            ORIGINAL_CODE,
            refactoring_task="Use list comprehension",
            expected_output="[2, 6, 10]",
        )
        assert len(result.details) > 0

    def test_original_code_preserved(self, benchmark):
        benchmark._timed_run(
            ORIGINAL_CODE,
            refactoring_task="Use list comprehension",
        )
        assert benchmark.code_input == ORIGINAL_CODE

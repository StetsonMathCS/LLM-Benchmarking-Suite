"""
End-to-end tests for the CodeGenerationBenchmark task.
"""
import pytest
from benchmarks.tasks.code_generation import CodeGenerationBenchmark
from core.base import BenchmarkResult, BenchmarkStatus
from tests.conftest import FakeProvider


GENERATED_ADD = """\
def add(a, b):
    return a + b
"""

TEST_CODE = """\
assert add(1, 2) == 3
assert add(0, 0) == 0
assert add(-1, 1) == 0
"""


@pytest.fixture
def benchmark():
    provider = FakeProvider(response_content=GENERATED_ADD)
    return CodeGenerationBenchmark(code_language="python", provider=provider)


@pytest.fixture
def benchmark_error():
    provider = FakeProvider(error="timeout")
    return CodeGenerationBenchmark(code_language="python", provider=provider)


# ── Prompt building ───────────────────────────────────────────────────────

class TestCodeGenerationPrompt:
    def test_build_prompt_contains_input(self, benchmark):
        prompt = benchmark.build_prompt("python", "Write a function that adds two numbers")
        assert "adds two numbers" in prompt

    def test_build_prompt_contains_language(self, benchmark):
        prompt = benchmark.build_prompt("python", "Write code")
        assert "python" in prompt.lower() or "Python" in prompt


# ── Full run ──────────────────────────────────────────────────────────────

class TestCodeGenerationRun:
    def test_successful_run_without_tests(self, benchmark):
        result = benchmark._timed_run(
            "Write a function that adds two numbers",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.status in (BenchmarkStatus.PASSED, BenchmarkStatus.FAILED)
        assert result.combined_score is not None

    def test_successful_run_with_tests(self, benchmark):
        result = benchmark._timed_run(
            "Write a function that adds two numbers",
            test=TEST_CODE,
            entry_point="add",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.combined_score is not None

    def test_error_response(self, benchmark_error):
        result = benchmark_error._timed_run(
            "Write a function that adds two numbers",
        )
        assert result.status == BenchmarkStatus.ERROR

    def test_non_python_skips_completion_tests(self):
        provider = FakeProvider(response_content="function add(a, b) { return a + b; }")
        bm = CodeGenerationBenchmark(code_language="javascript", provider=provider)
        result = bm._timed_run("Write add function")
        assert isinstance(result, BenchmarkResult)

    def test_dimensions_in_details(self, benchmark):
        result = benchmark._timed_run("Write add function")
        assert len(result.details) > 0

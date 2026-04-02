"""
End-to-end tests for the PartialTransformBenchmark task.
"""
import pytest
from benchmarks.tasks.partial_transform import PartialTransformBenchmark
from core.base import BenchmarkResult, BenchmarkStatus
from tests.conftest import FakeProvider


ORIGINAL_CODE = """\
result = []
for i in range(10):
    result.append(i * 2)
print(result)
"""

TRANSFORMED_CODE = """\
result = [i * 2 for i in range(10)]
print(result)
"""


@pytest.fixture
def benchmark():
    provider = FakeProvider(response_content=TRANSFORMED_CODE)
    return PartialTransformBenchmark(code_language="python", provider=provider)


@pytest.fixture
def benchmark_error():
    provider = FakeProvider(error="internal error")
    return PartialTransformBenchmark(code_language="python", provider=provider)


# ── Prompt building ───────────────────────────────────────────────────────

class TestPartialTransformPrompt:
    def test_build_prompt_contains_code(self, benchmark):
        prompt = benchmark.build_prompt(
            "python",
            ORIGINAL_CODE,
            transform_from="for loop",
            transform_to="list comprehension",
        )
        assert "result" in prompt

    def test_build_prompt_contains_transforms(self, benchmark):
        prompt = benchmark.build_prompt(
            "python",
            ORIGINAL_CODE,
            transform_from="for loop",
            transform_to="list comprehension",
        )
        assert "for loop" in prompt
        assert "list comprehension" in prompt

    def test_build_prompt_requires_transform_kwargs(self, benchmark):
        with pytest.raises(KeyError):
            benchmark.build_prompt("python", ORIGINAL_CODE)


# ── Full run ──────────────────────────────────────────────────────────────

class TestPartialTransformRun:
    def test_successful_run(self, benchmark):
        result = benchmark._timed_run(
            ORIGINAL_CODE,
            transform_from="for",
            transform_to="list comprehension",
            expected_output="[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.status in (BenchmarkStatus.PASSED, BenchmarkStatus.FAILED)
        assert result.combined_score is not None
        assert 0.0 <= result.combined_score <= 1.0

    def test_run_with_ast_verification(self, benchmark):
        result = benchmark._timed_run(
            ORIGINAL_CODE,
            transform_from="for",
            transform_to="list comprehension",
            verify_ast_nodes_removed="For",
            verify_ast_nodes_added="ListComp",
            expected_output="[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.combined_score is not None

    def test_error_response(self, benchmark_error):
        result = benchmark_error._timed_run(
            ORIGINAL_CODE,
            transform_from="for",
            transform_to="while",
        )
        assert result.status == BenchmarkStatus.ERROR

    def test_dimensions_evaluated(self, benchmark):
        result = benchmark._timed_run(
            ORIGINAL_CODE,
            transform_from="for",
            transform_to="list comprehension",
        )
        assert len(result.details) > 0

    def test_original_code_stored(self, benchmark):
        benchmark._timed_run(
            ORIGINAL_CODE,
            transform_from="for",
            transform_to="list comprehension",
        )
        assert benchmark.code_input == ORIGINAL_CODE

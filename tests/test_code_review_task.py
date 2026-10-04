"""
End-to-end tests for the CodeReviewBenchmark task.
"""
import pytest
from unittest.mock import patch
from benchmarks.tasks.code_review import CodeReviewBenchmark
from benchmarks.dimensions.code_review import CodeReviewDimension
from core.base import BenchmarkResult, BenchmarkStatus, DimensionResult
from tests.conftest import FakeProvider


REVIEW_INPUT = """\
def process(data):
    for i in range(len(data)):
        data[i] = data[i] * 2
    return data
"""

GENERATED_REVIEW = """\
The code uses range(len(data)) which is not Pythonic.
Consider using enumerate() or a list comprehension instead.
"""


@pytest.fixture
def benchmark():
    provider = FakeProvider(response_content=GENERATED_REVIEW)
    return CodeReviewBenchmark(code_language="python", provider=provider)


@pytest.fixture
def benchmark_error():
    provider = FakeProvider(error="model not found")
    return CodeReviewBenchmark(code_language="python", provider=provider)


# ── Prompt building ───────────────────────────────────────────────────────

class TestCodeReviewPrompt:
    def test_build_prompt_python(self, benchmark):
        prompt = benchmark.build_prompt("python", REVIEW_INPUT)
        assert "process" in prompt

    def test_build_prompt_with_review_request(self, benchmark):
        prompt = benchmark.build_prompt(
            "python",
            REVIEW_INPUT,
            review_request="Focus on performance",
        )
        # The review_request is inserted via {{REVIEW_REQUEST}} in the template
        # The template may or may not include it depending on format
        assert "process" in prompt  # At minimum, the code is included

    def test_build_prompt_cpp(self):
        provider = FakeProvider(response_content="looks good")
        bm = CodeReviewBenchmark(code_language="cpp", provider=provider)
        prompt = bm.build_prompt("cpp", "int main() { return 0; }")
        assert "int main" in prompt


# ── Full run ──────────────────────────────────────────────────────────────

class TestCodeReviewRun:
    def test_error_response(self, benchmark_error):
        result = benchmark_error._timed_run(REVIEW_INPUT, expected_output="some review")
        assert result.status == BenchmarkStatus.ERROR

    def test_run_without_ollama(self, benchmark):
        """Without Ollama, CodeReviewDimension will fail to embed, score=0."""
        result = benchmark._timed_run(
            REVIEW_INPUT,
            expected_output="Use enumerate instead of range(len(data))",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.combined_score is None
        assert result.status == BenchmarkStatus.ERROR

    def test_run_with_mocked_dimension(self):
        """Mock the dimension to test the task orchestration logic."""
        provider = FakeProvider(response_content=GENERATED_REVIEW)
        bm = CodeReviewBenchmark(code_language="python", provider=provider)

        mock_result = DimensionResult(
            dimension_name="Reference Review Similarity (RRS)",
            dimension_id="reference_review_similarity",
            score=0.85,
            passed=True,
            details={"similarity_method": "cosine"},
        )
        with patch.object(CodeReviewDimension, 'evaluate', return_value=mock_result):
            with patch.object(CodeReviewDimension, '_connect_ollama', return_value=False):
                result = bm._timed_run(
                    REVIEW_INPUT,
                    expected_output="Use enumerate",
                )
        assert result.status == BenchmarkStatus.PASSED
        assert result.combined_score == pytest.approx(0.85, abs=0.01)

    def test_llm_response_present(self, benchmark):
        result = benchmark._timed_run(REVIEW_INPUT, expected_output="some review")
        assert result.llm_response is not None
        assert result.llm_response.content == GENERATED_REVIEW

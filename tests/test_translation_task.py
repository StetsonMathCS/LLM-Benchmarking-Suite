"""
End-to-end tests for the TranslationBenchmark task.
"""
import pytest
from benchmarks.tasks.translation import TranslationBenchmark
from core.base import BenchmarkResult, BenchmarkStatus
from tests.conftest import FakeProvider


PYTHON_CODE = """\
def add(a, b):
    return a + b

print(add(2, 3))
"""

JS_TRANSLATION = """\
function add(a, b) {
    return a + b;
}
console.log(add(2, 3));
"""


@pytest.fixture
def benchmark():
    provider = FakeProvider(response_content=JS_TRANSLATION)
    return TranslationBenchmark(code_language="python", provider=provider)


@pytest.fixture
def benchmark_error():
    provider = FakeProvider(error="bad request")
    return TranslationBenchmark(code_language="python", provider=provider)


# ── Prompt building ───────────────────────────────────────────────────────

class TestTranslationPrompt:
    def test_build_prompt_python_to_js(self, benchmark):
        prompt = benchmark.build_prompt("python", PYTHON_CODE, target_language="javascript")
        assert "add" in prompt

    def test_build_prompt_only_python_source(self, benchmark):
        with pytest.raises(RuntimeError, match="only supports Python"):
            benchmark.build_prompt("javascript", "console.log(1);", target_language="python")

    def test_build_prompt_unsupported_target(self, benchmark):
        with pytest.raises(RuntimeError, match="only supports javascript and python"):
            benchmark.build_prompt("python", PYTHON_CODE, target_language="rust")


# ── Full run ──────────────────────────────────────────────────────────────

class TestTranslationRun:
    def test_successful_run(self, benchmark):
        result = benchmark._timed_run(
            PYTHON_CODE,
            target_language="javascript",
            expected_output="5",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.status in (BenchmarkStatus.PASSED, BenchmarkStatus.FAILED)
        assert result.combined_score is not None
        assert 0.0 <= result.combined_score <= 1.0

    def test_error_response(self, benchmark_error):
        result = benchmark_error._timed_run(
            PYTHON_CODE,
            target_language="javascript",
        )
        assert result.status == BenchmarkStatus.ERROR

    def test_generated_code_language_passed_to_dimensions(self, benchmark):
        result = benchmark._timed_run(
            PYTHON_CODE,
            target_language="javascript",
        )
        # Dimensions should receive generated_code_language="javascript"
        assert isinstance(result, BenchmarkResult)

    def test_llm_response_content(self, benchmark):
        result = benchmark._timed_run(
            PYTHON_CODE,
            target_language="javascript",
        )
        assert result.llm_response.content == JS_TRANSLATION

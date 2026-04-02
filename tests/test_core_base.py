"""
End-to-end tests for core base classes and dataclasses.
"""
import pytest
from core.base import (
    BenchmarkStatus,
    LLMResponse,
    ModelConfig,
    BenchmarkResult,
    DimensionResult,
)


# ── LLMResponse ───────────────────────────────────────────────────────────

class TestLLMResponse:
    def test_success_with_content(self):
        r = LLMResponse(content="hello", model="m", provider="p")
        assert r.success is True

    def test_failure_with_error(self):
        r = LLMResponse(content="", model="m", provider="p", error="boom")
        assert r.success is False

    def test_failure_with_empty_content(self):
        r = LLMResponse(content="", model="m", provider="p")
        assert r.success is False

    def test_default_token_counts(self):
        r = LLMResponse(content="x", model="m", provider="p")
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0


# ── ModelConfig ───────────────────────────────────────────────────────────

class TestModelConfig:
    def test_required_fields(self):
        c = ModelConfig(provider="openai", model_name="gpt-4")
        assert c.provider == "openai"
        assert c.model_name == "gpt-4"

    def test_optional_fields_default_none(self):
        c = ModelConfig(provider="openai", model_name="gpt-4")
        assert c.api_key is None
        assert c.base_url is None
        assert c.system_prompt is None

    def test_extra_params_default_empty(self):
        c = ModelConfig(provider="openai", model_name="gpt-4")
        assert c.extra_params == {}


# ── BenchmarkResult ───────────────────────────────────────────────────────

class TestBenchmarkResult:
    def test_to_dict(self):
        r = BenchmarkResult(
            benchmark_name="test",
            status=BenchmarkStatus.PASSED,
            combined_score=0.85,
        )
        d = r.to_dict()
        assert d["benchmark"] == "test"
        assert d["status"] == "passed"
        assert d["combined_score"] == 0.85

    def test_default_fields(self):
        r = BenchmarkResult(
            benchmark_name="test",
            status=BenchmarkStatus.PENDING,
        )
        assert r.combined_score is None
        assert r.details == {}
        assert r.issues_found == {}
        assert r.duration_s == 0.0


# ── DimensionResult ──────────────────────────────────────────────────────

class TestDimensionResult:
    def test_defaults(self):
        r = DimensionResult(dimension_name="test", score=0.5)
        assert r.passed is True
        assert r.details == {}
        assert r.issues == []

    def test_custom_values(self):
        r = DimensionResult(
            dimension_name="test",
            score=0.0,
            passed=False,
            details={"error": "fail"},
            issues=["issue1"],
        )
        assert r.passed is False
        assert r.score == 0.0
        assert len(r.issues) == 1


# ── BenchmarkStatus ──────────────────────────────────────────────────────

class TestBenchmarkStatus:
    def test_all_statuses_exist(self):
        assert BenchmarkStatus.PENDING.value == "pending"
        assert BenchmarkStatus.RUNNING.value == "running"
        assert BenchmarkStatus.PASSED.value == "passed"
        assert BenchmarkStatus.FAILED.value == "failed"
        assert BenchmarkStatus.ERROR.value == "error"
        assert BenchmarkStatus.SKIPPED.value == "skipped"

    def test_status_count(self):
        assert len(BenchmarkStatus) == 6

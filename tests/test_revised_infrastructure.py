import json
from pathlib import Path

import pytest

from core.base import BenchmarkResult, BenchmarkStatus, DimensionResult, LLMResponse, ModelConfig
from core.scoring import ScoringEngine
from facets.config import load_model_registry, merge_cli, resolve_model, validate_resolved
from facets.migration import LegacyParseError, parse_legacy_repr
from facets.scoring_profile import load_profile
from providers.openai.provider import _text_content


def test_structured_schema_serializes_without_default_str():
    result = BenchmarkResult(
        benchmark_name="x",
        status=BenchmarkStatus.PASSED,
        combined_score=1.0,
        details={"reference_test_success": DimensionResult("Reference Test Success (RTS)", 1.0)},
        llm_response=LLMResponse("code", "model", "fake"),
    )
    encoded = json.dumps(result.to_dict(), allow_nan=False)
    assert "DimensionResult(" not in encoded
    assert '"content": "code"' in encoded


def test_weight_validation_rejects_point_nine(tmp_path):
    profile = (Path(__file__).resolve().parents[1] / "profiles" / "revised-v1.yaml").read_text()
    profile = profile.replace("code_generation: 0.27", "code_generation: 0.17")
    path = tmp_path / "bad.yaml"
    path.write_text(profile)
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_profile(str(path))


def test_cli_precedence_preserves_omitted_values():
    config = {"temperature": 0.7, "max_tokens": 100}
    assert merge_cli(config, {"temperature": None, "max_tokens": 200}) == {"temperature": 0.7, "max_tokens": 200}


def test_provider_credentials_never_fall_back(monkeypatch):
    registry = {
        "models": {},
        "provider_capabilities": {
            "openai": {"credential_env": "OPENAI_API_KEY", "supported_options": []},
            "anthropic": {"credential_env": "ANTHROPIC_API_KEY", "supported_options": []},
        },
    }
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec = resolve_model(None, "anthropic", "exact-id", registry)
    issues = validate_resolved({"tasks": ["code_generation"], "language": "python", "profile": "revised-v1"}, spec, registry)
    assert any(issue["detail"] == "ANTHROPIC_API_KEY" for issue in issues)


def test_short_aliases_resolve_to_one_canonical_model():
    registry = {
        "aliases": {"sonnet": "claude-sonnet-4-6", "short": "sonnet"},
        "models": {"claude-sonnet-4-6": {"provider": "anthropic", "model": "claude-sonnet-4-6"}},
    }
    canonical = resolve_model("claude-sonnet-4-6", None, None, registry)
    short = resolve_model("sonnet", None, None, registry)
    nested = resolve_model("short", None, None, registry)

    assert short == {**canonical, "requested_alias": "sonnet"}
    assert nested["alias"] == "claude-sonnet-4-6"
    assert nested["model"] == "claude-sonnet-4-6"
    # The canonical key must stay the model identity so cohorts cannot fork on spelling.
    assert resolve_model("claude-sonnet-4-6", None, None, registry).get("requested_alias") is None


def test_unknown_and_circular_aliases_are_refused():
    registry = {"aliases": {"loop": "loop"}, "models": {"real": {"provider": "ollama", "model": "real"}}}
    with pytest.raises(ValueError, match="alias cycle"):
        resolve_model("loop", None, None, registry)
    with pytest.raises(ValueError, match="unknown model alias"):
        resolve_model("missing", None, None, registry)


def test_shipped_registry_aliases_all_point_at_real_models():
    registry = load_model_registry()
    models = registry["models"]
    assert registry["aliases"]
    for short, canonical in registry["aliases"].items():
        assert canonical in models, f"{short} -> {canonical} is not a registry model"
        assert resolve_model(short, None, None, registry)["alias"] == canonical


def test_multiblock_text_extraction_and_truncation_metadata():
    assert _text_content([{"text": "first"}, {"text": " second"}]) == "first second"
    response = LLMResponse("x", "m", "p", stop_reason="length", truncated=True)
    assert response.to_dict()["truncated"] is True


def test_restricted_legacy_parser_never_executes_calls():
    parsed = parse_legacy_repr("DimensionResult(dimension_name='Code Generation Tests', score=1.0, passed=True, details={}, issues=[])")
    assert parsed["score"] == 1.0
    with pytest.raises(LegacyParseError):
        parse_legacy_repr("__import__('os').system('false')")


def test_actual_functional_pass_at_k_is_not_composite_threshold():
    records = []
    for index, (score, correct) in enumerate([(0.9, False), (0.1, True)]):
        records.append(BenchmarkResult(
            "code_generation", BenchmarkStatus.PASSED, score,
            metadata={"task_name": "code_generation", "record_id": "p", "sample_index": index, "functional_correct": correct},
        ))
    report = ScoringEngine(records, num_samples=2, pass_k_values=[1, 2]).compute()
    assert report.pass_at_k["code_generation"][1] == 0.5
    assert report.pass_at_k["code_generation"][2] == 1.0


def test_insufficient_pass_k_is_unavailable():
    record = BenchmarkResult(
        "code_generation", BenchmarkStatus.PASSED, 1.0,
        metadata={"task_name": "code_generation", "record_id": "p", "functional_correct": True},
    )
    report = ScoringEngine([record], num_samples=1, pass_k_values=[5]).compute()
    assert report.pass_at_k["code_generation"][5] is None


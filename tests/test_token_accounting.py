import json

import pytest

from benchmarks.dimensions.code_review import ReferenceReviewSimilarityDimension
from core.base import BaseProvider, LLMResponse, ModelConfig, usage_fields
from facets.runner import execute_run
from facets.usage import (
    LEDGER_RELATIVE_PATH,
    STATUS_FAILED,
    normalize_saved_response,
    normalize_usage,
    summarize_usage,
    write_ledger_exports,
)


class _Usage:
    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)


def _entry(**overrides):
    base = {
        "request_id": "r1",
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "purpose": "generation",
        "provider": "anthropic",
        "model": "claude",
        "task": "code_generation",
        "record_id": "rec-1",
        "sample_index": 0,
        "attempt": 1,
        "response_status": "success",
        "usage_origin": "measured_here",
        "usage": {
            "measured": True,
            "counted_input_tokens": 100,
            "counted_output_tokens": 20,
            "usage_source": "provider_reported",
        },
        "counted_total_tokens": 120,
    }
    base.update(overrides)
    return base


def test_anthropic_cache_tokens_are_additional_to_input():
    usage = normalize_usage("anthropic", {
        "input_tokens": 100,
        "output_tokens": 40,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 60,
    })
    assert usage.counted_input_tokens == 100 + 900 + 60
    assert usage.cache_accounting == "additional_to_input"
    assert usage.counted_output_tokens == 40
    assert usage.counted_total == 100 + 900 + 60 + 40


def test_anthropic_without_cache_reports_no_cache_accounting():
    usage = normalize_usage("anthropic", {"input_tokens": 100, "output_tokens": 40})
    assert usage.cache_accounting == "not_reported"
    assert usage.counted_input_tokens == 100


def test_openai_reasoning_and_cached_tokens_are_subsets():
    responses = normalize_usage("openai_responses", {
        "input_tokens": 500,
        "output_tokens": 200,
        "total_tokens": 700,
        "output_tokens_details": {"reasoning_tokens": 120},
    })
    assert responses.counted_input_tokens == 500
    assert responses.counted_output_tokens == 200
    assert responses.reasoning_tokens == 120
    assert responses.reasoning_accounting == "included_in_output"
    assert responses.counted_total == 700

    chat = normalize_usage("openai", {
        "prompt_tokens": 300,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 250},
    })
    assert chat.counted_input_tokens == 300
    assert chat.cache_read_tokens == 250
    assert chat.cache_accounting == "included_in_input"


def test_openai_chat_adapter_selected_from_field_names():
    usage = normalize_usage("openai", {"prompt_tokens": 10, "completion_tokens": 5})
    assert usage.counted_input_tokens == 10
    assert usage.counted_output_tokens == 5


def test_ollama_usage_reads_eval_counts():
    usage = normalize_usage("ollama", {"prompt_eval_count": 42, "eval_count": 7})
    assert usage.counted_input_tokens == 42
    assert usage.counted_output_tokens == 7
    assert usage.cache_accounting == "not_reported"


def test_unknown_usage_is_not_zero():
    usage = normalize_usage("openai", {})
    assert usage.measured is False
    assert usage.counted_input_tokens is None
    assert normalize_usage("nonesuch", {"prompt_tokens": 5}).measured is False


def test_saved_response_without_usage_stays_unknown():
    usage = normalize_saved_response("openai", {"content": "x"})
    assert usage.measured is False
    legacy = normalize_saved_response("openai", {"prompt_tokens": 10, "completion_tokens": 2, "latency_ms": 5})
    assert legacy.measured is True
    assert legacy.counted_input_tokens == 10
    assert legacy.source == "saved_response_fields"


def test_usage_fields_flattens_sdk_objects():
    details = _Usage(cached_tokens=12, audio_tokens=0)
    payload = usage_fields(
        _Usage(input_tokens=100, output_tokens=20, output_tokens_details=details),
        ("input_tokens", "output_tokens", "output_tokens_details", "missing"),
    )
    assert payload == {
        "input_tokens": 100,
        "output_tokens": 20,
        "output_tokens_details": {"cached_tokens": 12, "audio_tokens": 0},
    }
    assert usage_fields(None, ("input_tokens",)) == {}


def test_duplicate_requests_counted_once():
    summary = summarize_usage([_entry(), _entry(recorded_at="2026-01-01T00:01:00+00:00")])
    assert summary["requests_total"] == 1
    assert summary["duplicate_entries_suppressed"] == 1
    assert summary["counted_total_tokens"] == 120


def test_unknown_entries_reduce_exactness_without_zero_filling():
    entries = [
        _entry(),
        _entry(request_id="r2", usage={"measured": False, "counted_input_tokens": None, "counted_output_tokens": None}, counted_total_tokens=None),
    ]
    summary = summarize_usage(entries)
    assert summary["requests_total"] == 2
    assert summary["requests_with_unknown_usage"] == 1
    assert summary["known_coverage"] == 0.5
    assert summary["usage_totals_are_exact"] is False
    assert summary["known_subtotal_only"] is True
    assert summary["counted_total_tokens"] == 120


def test_grouping_and_purpose_split():
    entries = [
        _entry(),
        _entry(request_id="r2", purpose="embedding", task=None, counted_total_tokens=30,
               usage={"measured": True, "counted_input_tokens": 30, "counted_output_tokens": 0}),
        _entry(request_id="r3", attempt=2, response_status=STATUS_FAILED, counted_total_tokens=5,
               usage={"measured": True, "counted_input_tokens": 5, "counted_output_tokens": 0}),
    ]
    summary = summarize_usage(entries)
    assert summary["successful_generation"]["counted_total_tokens"] == 120
    assert summary["by_purpose"]["embedding"]["counted_total_tokens"] == 30
    assert summary["retry_and_failure_overhead"]["counted_total_tokens"] == 35
    assert summary["by_record"]["code_generation/rec-1/0"]["counted_total_tokens"] == 125
    assert summary["by_task"]["code_generation"]["requests_total"] == 2


def test_summary_since_isolates_new_usage():
    entries = [
        _entry(recorded_at="2026-01-01T00:00:00+00:00"),
        _entry(request_id="r2", recorded_at="2026-01-02T00:00:00+00:00", counted_total_tokens=7,
               usage={"measured": True, "counted_input_tokens": 7, "counted_output_tokens": 0}),
    ]
    summary = summarize_usage(entries, since="2026-01-01T12:00:00+00:00")
    assert summary["counted_total_tokens"] == 127
    assert summary["new_since_resume"]["counted_total_tokens"] == 7


class _UsageProvider(BaseProvider):
    def __init__(self, responses):
        super().__init__(ModelConfig("ollama", "fake"))
        self.responses = list(responses)
        self.calls = 0

    def connect(self):
        return True

    def is_available(self):
        return True

    def complete(self, prompt, system_prompt=None):
        self.calls += 1
        content, usage, error = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return LLMResponse(
            content=content,
            model="fake",
            provider="ollama",
            prompt_tokens=usage.get("prompt_eval_count", 0),
            completion_tokens=usage.get("eval_count", 0),
            usage=usage,
            error=error,
            latency_ms=12.5,
        )


def _config(tmp_path):
    return {
        "profile": "revised-v1",
        "language": "python",
        "tasks": ["code_generation"],
        "samples": 1,
        "limit": 1,
        "output_dir": str(tmp_path),
        "execution": {"backend": "local"},
        "timeouts": {"evaluator_s": 2},
        "retries": {"attempts": 2, "initial_backoff_s": 0, "max_backoff_s": 0},
    }


def _model():
    return {"alias": "fake", "provider": "ollama", "model": "fake", "credential_env": None, "availability": "test"}


def test_run_ledger_records_provider_usage(tmp_path):
    provider = _UsageProvider([("x = 1\n", {"prompt_eval_count": 100, "eval_count": 20}, None)])
    run_dir = tmp_path / "run"
    execute_run(_config(tmp_path), _model(), run_dir, provider_override=provider)

    entries = [json.loads(line) for line in (run_dir / LEDGER_RELATIVE_PATH).read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["purpose"] == "generation"
    assert entries[0]["counted_total_tokens"] == 120
    assert entries[0]["usage"]["measured"] is True
    assert entries[0]["task"] == "code_generation"

    usage_summary = json.loads((run_dir / "usage" / "summary.json").read_text())
    assert usage_summary["usage_totals_are_exact"] is True
    assert usage_summary["counted_total_tokens"] == 120
    assert usage_summary["by_purpose"]["generation"]["requests_total"] == 1

    report = json.loads((run_dir / "summary.json").read_text())
    assert report["token_usage"]["generation"]["counted_total_tokens"] == 120
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["token_usage"]["totals_are_exact"] is True


def test_failed_attempts_are_recorded_separately(tmp_path):
    provider = _UsageProvider([
        ("", {}, "rate limit exceeded"),
        ("x = 1\n", {"prompt_eval_count": 10, "eval_count": 2}, None),
    ])
    run_dir = tmp_path / "run"
    execute_run(_config(tmp_path), _model(), run_dir, provider_override=provider)

    entries = [json.loads(line) for line in (run_dir / LEDGER_RELATIVE_PATH).read_text().splitlines()]
    assert [(entry["attempt"], entry["response_status"]) for entry in entries] == [
        (1, STATUS_FAILED), (2, "success")
    ]
    usage_summary = json.loads((run_dir / "usage" / "summary.json").read_text())
    assert usage_summary["requests_with_unknown_usage"] == 1
    assert usage_summary["usage_totals_are_exact"] is False
    assert usage_summary["retry_and_failure_overhead"]["requests_total"] == 1
    assert usage_summary["successful_generation"]["counted_total_tokens"] == 12


def test_resume_records_inherited_usage_without_double_counting(tmp_path):
    provider = _UsageProvider([("x = 1\n", {"prompt_eval_count": 40, "eval_count": 4}, None)])
    run_dir = tmp_path / "run"
    execute_run(_config(tmp_path), _model(), run_dir, provider_override=provider)

    second = _UsageProvider([("y = 2\n", {"prompt_eval_count": 999, "eval_count": 999}, None)])
    execute_run(_config(tmp_path), _model(), run_dir, resume=True, provider_override=second)
    assert second.calls == 0

    entries = (run_dir / LEDGER_RELATIVE_PATH).read_text().splitlines()
    assert len(entries) == 1
    usage_summary = json.loads((run_dir / "usage" / "summary.json").read_text())
    assert usage_summary["counted_total_tokens"] == 44
    assert usage_summary["duplicate_entries_suppressed"] == 0


def test_legacy_saved_response_usage_is_recovered(tmp_path):
    provider = _UsageProvider([("x = 1\n", {}, None)])
    run_dir = tmp_path / "run"
    execute_run(_config(tmp_path), _model(), run_dir, provider_override=provider)

    saved = next((run_dir / "responses").glob("*.json"))
    payload = json.loads(saved.read_text())
    payload["response"].pop("usage")
    payload["response"]["prompt_tokens"] = 15
    payload["response"]["completion_tokens"] = 3
    saved.write_text(json.dumps(payload))
    (run_dir / "usage" / "ledger.jsonl").unlink()
    (run_dir / "results.jsonl").unlink()

    execute_run(_config(tmp_path), _model(), run_dir, resume=True, provider_override=_UsageProvider([("", {}, None)]))
    usage_summary = json.loads((run_dir / "usage" / "summary.json").read_text())
    assert usage_summary["counted_total_tokens"] == 18
    assert usage_summary["inherited_requests"] == 1
    assert usage_summary["by_purpose"]["generation"]["requests_with_known_usage"] == 1


def test_embedding_sink_reports_usage_without_extra_requests():
    recorded = []
    ReferenceReviewSimilarityDimension.USAGE_SINK = recorded.append
    dimension = ReferenceReviewSimilarityDimension.__new__(ReferenceReviewSimilarityDimension)
    dimension.EMBEDDING_MODEL = "nomic-embed-text:latest"

    dimension._report_usage({"prompt_eval_count": 64}, 8.5, "success")
    dimension._report_usage(None, None, "failed")
    ReferenceReviewSimilarityDimension.USAGE_SINK = None

    assert [entry["purpose"] for entry in recorded] == ["embedding", "embedding"]
    assert recorded[0]["raw_usage"] == {"prompt_eval_count": 64}
    assert recorded[1]["status"] == "failed"


def test_usage_context_is_thread_local():
    from facets.usage import UsageContext, get_context

    results = {}

    def worker():
        results["worker"] = get_context()

    with UsageContext(task="translation", record_id="r"):
        results["main"] = get_context()
        thread = pytest.importorskip("threading").Thread(target=worker)
        thread.start()
        thread.join()
    assert results["main"]["task"] == "translation"
    assert results["worker"] == {}


def test_write_ledger_exports_csv(tmp_path):
    from facets.usage import LedgerScope, UsageLedger

    ledger = UsageLedger(tmp_path / "run", LedgerScope(run_id="run-1"))
    ledger.record(_entry())
    summary = write_ledger_exports(ledger)

    assert summary["requests_total"] == 1
    rows = (tmp_path / "run" / "usage" / "requests.csv").read_text().splitlines()
    assert rows[0].startswith("request_id,recorded_at,purpose")
    assert "r1" in rows[1]


def test_ledger_survives_torn_final_line(tmp_path):
    from facets.usage import iter_ledger

    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(_entry()) + "\n" + '{"request_id": "r2", "trunc')
    assert [entry["request_id"] for entry in iter_ledger(path)] == ["r1"]


def test_failed_attempt_without_usage_is_recorded_as_unknown(tmp_path):
    provider = _UsageProvider([("", {}, "connection reset")])
    run_dir = tmp_path / "run"
    execute_run(_config(tmp_path), _model(), run_dir, provider_override=provider)
    entries = [json.loads(line) for line in (run_dir / LEDGER_RELATIVE_PATH).read_text().splitlines()]
    assert entries[0]["usage"]["measured"] is False
    assert entries[0]["error"]
    assert entries[0]["counted_total_tokens"] is None

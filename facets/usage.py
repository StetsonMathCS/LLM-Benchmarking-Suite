"""Provider-reported token accounting for every model request FACETS makes.

Only provider-reported usage is counted.  No request is issued to obtain token
counts and no local tokenizer estimate is presented as measured usage.

Counting semantics
------------------
``input_tokens``   provider-reported input tokens for the request, including
                   system prompts and other billed request context exactly as
                   the provider counts them.
``output_tokens``  provider-reported generated tokens.
``total_tokens``   provider-reported total when available, otherwise the sum
                   of the counted input/output values implied by the provider's
                   documented accounting rule.
``counted_*``      the values summed for totals; cache/reasoning fields are
                   never added twice.

Cache and reasoning accounting differs per provider and is recorded per entry:

* Anthropic ``cache_creation_input_tokens`` and ``cache_read_input_tokens`` are
  **additional** categories: total input is their sum with ``input_tokens``.
* OpenAI ``input_tokens_details.cached_tokens``/``cache_write_tokens`` and
  ``output_tokens_details.reasoning_tokens`` are **subsets** of
  ``input_tokens``/``output_tokens``.
* Ollama reports no cache or reasoning breakdown.

Requests without provider usage are recorded with ``usage_source:
unavailable`` and are never counted as zero.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from facets.config import fingerprint
from facets.runstore import atomic_json

LEDGER_SCHEMA_VERSION = "1"
LEDGER_RELATIVE_PATH = Path("usage") / "ledger.jsonl"
LEDGER_CSV_COLUMNS = (
    "request_id", "recorded_at", "purpose", "provider", "model", "task", "record_id",
    "sample_index", "attempt", "attempts_total", "response_status", "error_present",
    "input_tokens", "output_tokens", "total_tokens", "usage_known", "usage_source",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "cache_accounting",
    "reasoning_accounting", "latency_ms", "usage_origin", "run_directory", "queue_id",
    "queue_item_id",
)

PURPOSE_GENERATION = "generation"
PURPOSE_EMBEDDING = "embedding"

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_TRUNCATED = "truncated"
STATUS_CANCELLED = "cancelled"

SOURCE_PROVIDER = "provider_reported"
SOURCE_SAVED_FIELDS = "saved_response_fields"
SOURCE_UNAVAILABLE = "unavailable"

INPUT_SUBSET = "included_in_input"
INPUT_ADDITIONAL = "additional_to_input"
OUTPUT_SUBSET = "included_in_output"
NOT_REPORTED = "not_reported"

TOKEN_SEMANTICS = {
    "input_tokens": "provider-reported request input tokens, including system prompts and other billed request context",
    "output_tokens": "provider-reported generated tokens",
    "total_tokens": "provider-reported total when available, otherwise counted input plus counted output",
    "counted_input_tokens": "input tokens summed for totals under the provider's documented accounting rule",
    "counted_output_tokens": "output tokens summed for totals under the provider's documented accounting rule",
    "cache_read_tokens": "tokens served from the provider prompt cache",
    "cache_write_tokens": "tokens written to the provider prompt cache",
    "reasoning_tokens": "provider-reported reasoning tokens",
    "usage_source": f"{SOURCE_PROVIDER} | {SOURCE_SAVED_FIELDS} | {SOURCE_UNAVAILABLE}",
    "coverage": "attempts with known usage versus total attempts; unknown usage is never counted as zero",
    "exclusions": "code execution, static analysis, plotting, and YAML validation consume no model tokens",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _first_present(payload: dict, *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_int(payload: dict, container: str, key: str) -> int | None:
    details = payload.get(container)
    if isinstance(details, dict):
        return _as_int(details.get(key))
    return None


@dataclass(frozen=True)
class NormalizedUsage:
    """Provider usage reduced to one comparable accounting rule."""

    measured: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    counted_input_tokens: int | None = None
    counted_output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_accounting: str = NOT_REPORTED
    reasoning_accounting: str = NOT_REPORTED
    latency_ms: float | None = None
    source: str = SOURCE_UNAVAILABLE

    @property
    def counted_total(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.counted_input_tokens is None and self.counted_output_tokens is None:
            return None
        return (self.counted_input_tokens or 0) + (self.counted_output_tokens or 0)

    def with_latency(self, latency_ms: float | None) -> NormalizedUsage:
        if latency_ms is None or latency_ms == self.latency_ms:
            return self
        return replace(self, latency_ms=float(latency_ms))

    def to_dict(self) -> dict:
        return {
            "measured": self.measured,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "counted_input_tokens": self.counted_input_tokens,
            "counted_output_tokens": self.counted_output_tokens,
            "counted_total_tokens": self.counted_total,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_accounting": self.cache_accounting,
            "reasoning_accounting": self.reasoning_accounting,
            "latency_ms": self.latency_ms,
            "usage_source": self.source,
        }


UNMEASURED = NormalizedUsage()


def _openai_responses(payload: dict) -> NormalizedUsage:
    input_tokens = _as_int(_first_present(payload, "input_tokens"))
    output_tokens = _as_int(_first_present(payload, "output_tokens"))
    cached = _as_int(_nested_int(payload, "input_tokens_details", "cached_tokens"))
    cache_write = _as_int(_nested_int(payload, "input_tokens_details", "cache_write_tokens"))
    reasoning = _as_int(_nested_int(payload, "output_tokens_details", "reasoning_tokens"))
    total = _as_int(_first_present(payload, "total_tokens"))
    measured = any(value is not None for value in (input_tokens, output_tokens, total))
    return NormalizedUsage(
        measured=measured,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        counted_input_tokens=input_tokens,
        counted_output_tokens=output_tokens,
        cache_read_tokens=cached,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        cache_accounting=INPUT_SUBSET,
        reasoning_accounting=OUTPUT_SUBSET,
        latency_ms=_as_float(_first_present(payload, "latency_ms")),
        source=SOURCE_PROVIDER if measured else SOURCE_UNAVAILABLE,
    )


def _openai_chat(payload: dict) -> NormalizedUsage:
    input_tokens = _as_int(_first_present(payload, "prompt_tokens", "input_tokens"))
    output_tokens = _as_int(_first_present(payload, "completion_tokens", "output_tokens"))
    cached = _as_int(_nested_int(payload, "prompt_tokens_details", "cached_tokens"))
    if cached is None:
        cached = _as_int(_nested_int(payload, "input_tokens_details", "cached_tokens"))
    cache_write = _as_int(_nested_int(payload, "input_tokens_details", "cache_write_tokens"))
    reasoning = _as_int(_nested_int(payload, "completion_tokens_details", "reasoning_tokens"))
    if reasoning is None:
        reasoning = _as_int(_nested_int(payload, "output_tokens_details", "reasoning_tokens"))
    total = _as_int(_first_present(payload, "total_tokens"))
    measured = any(value is not None for value in (input_tokens, output_tokens, total))
    return NormalizedUsage(
        measured=measured,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        counted_input_tokens=input_tokens,
        counted_output_tokens=output_tokens,
        cache_read_tokens=cached,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        cache_accounting=INPUT_SUBSET,
        reasoning_accounting=OUTPUT_SUBSET,
        latency_ms=_as_float(_first_present(payload, "latency_ms")),
        source=SOURCE_PROVIDER if measured else SOURCE_UNAVAILABLE,
    )


def _anthropic(payload: dict) -> NormalizedUsage:
    input_tokens = _as_int(_first_present(payload, "input_tokens"))
    output_tokens = _as_int(_first_present(payload, "output_tokens"))
    cache_read = _as_int(_first_present(payload, "cache_read_input_tokens"))
    cache_write = _as_int(_first_present(payload, "cache_creation_input_tokens"))
    extra = [value for value in (cache_read, cache_write) if value is not None]
    counted_input = None
    if input_tokens is not None or extra:
        counted_input = (input_tokens or 0) + sum(extra)
    measured = any(value is not None for value in (input_tokens, output_tokens))
    return NormalizedUsage(
        measured=measured,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_as_int(_first_present(payload, "total_tokens")),
        counted_input_tokens=counted_input,
        counted_output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_accounting=OUTPUT_SUBSET if _nested_int(payload, "output_tokens_details", "reasoning_tokens") else NOT_REPORTED,
        cache_accounting=INPUT_ADDITIONAL if extra else NOT_REPORTED,
        latency_ms=_as_float(_first_present(payload, "latency_ms")),
        source=SOURCE_PROVIDER if measured else SOURCE_UNAVAILABLE,
    )


def _ollama(payload: dict) -> NormalizedUsage:
    input_tokens = _as_int(_first_present(payload, "prompt_eval_count", "prompt_tokens", "input_tokens"))
    output_tokens = _as_int(_first_present(payload, "eval_count", "completion_tokens", "output_tokens"))
    total = _as_int(_first_present(payload, "total_tokens"))
    measured = any(value is not None for value in (input_tokens, output_tokens, total))
    return NormalizedUsage(
        measured=measured,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        counted_input_tokens=input_tokens,
        counted_output_tokens=output_tokens,
        cache_accounting=NOT_REPORTED,
        reasoning_accounting=NOT_REPORTED,
        latency_ms=_as_float(_first_present(payload, "latency_ms")),
        source=SOURCE_PROVIDER if measured else SOURCE_UNAVAILABLE,
    )


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_ADAPTERS: dict[str, Callable[[dict], NormalizedUsage]] = {
    "openai_responses": _openai_responses,
    "openai_chat": _openai_chat,
    "anthropic": _anthropic,
    "ollama": _ollama,
}


def _openai_adapter(payload: dict) -> NormalizedUsage:
    if any(key in payload for key in ("prompt_tokens", "completion_tokens", "prompt_tokens_details", "completion_tokens_details")):
        return _openai_chat(payload)
    return _openai_responses(payload)


_ADAPTERS["openai"] = _openai_adapter


def normalize_usage(provider: str, payload: dict | None) -> NormalizedUsage:
    """Reduce one provider's reported usage fields to the shared accounting rule."""
    if not payload:
        return UNMEASURED
    adapter = _ADAPTERS.get(str(provider).lower())
    if adapter is None:
        return UNMEASURED
    return adapter(payload)


def _rebuilt(usage: NormalizedUsage, source: str) -> NormalizedUsage:
    return NormalizedUsage(
        measured=usage.measured,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        counted_input_tokens=usage.counted_input_tokens,
        counted_output_tokens=usage.counted_output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cache_accounting=usage.cache_accounting,
        reasoning_accounting=usage.reasoning_accounting,
        latency_ms=usage.latency_ms,
        source=source,
    )


def normalize_saved_response(provider: str, response: dict | None) -> NormalizedUsage:
    """Recover verifiable usage from a saved or legacy response record.

    Only fields actually stored on the response are recovered.  A missing
    prompt/completion count stays unknown; it is never inferred as zero.
    """
    if not response:
        return UNMEASURED
    raw = response.get("usage")
    if isinstance(raw, dict) and raw:
        usage = normalize_usage(provider, raw)
        if usage.measured:
            return _rebuilt(usage, SOURCE_PROVIDER)
    prompt_tokens = _as_int(response.get("prompt_tokens"))
    completion_tokens = _as_int(response.get("completion_tokens"))
    if prompt_tokens is None and completion_tokens is None:
        return UNMEASURED
    latency = _as_float(response.get("latency_ms"))
    return NormalizedUsage(
        measured=True,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        counted_input_tokens=prompt_tokens,
        counted_output_tokens=completion_tokens,
        cache_accounting=NOT_REPORTED,
        reasoning_accounting=NOT_REPORTED,
        latency_ms=latency,
        source=SOURCE_SAVED_FIELDS,
    )


# ---------------------------------------------------------------------------
# Record context (active record identity while a benchmark is evaluated)
# ---------------------------------------------------------------------------

_LOCAL = threading.local()


def _current_context() -> dict[str, Any]:
    return getattr(_LOCAL, "context", None) or {}


def set_context(**fields: Any) -> dict[str, Any]:
    previous = _current_context()
    _LOCAL.context = {key: value for key, value in fields.items() if value is not None}
    return previous


def get_context() -> dict[str, Any]:
    return dict(_current_context())


class UsageContext:
    """Bind request identity fields for the duration of one record's work.

    Context is thread-local so bounded generation concurrency cannot misattribute
    a request to the wrong record.
    """

    def __init__(self, **fields: Any):
        self.fields = fields
        self.previous: dict[str, Any] = {}

    def __enter__(self) -> dict[str, Any]:
        self.previous = _current_context()
        merged = {**self.previous, **{k: v for k, v in self.fields.items() if v is not None}}
        _LOCAL.context = merged
        return dict(merged)

    def __exit__(self, *exc_info: object) -> None:
        _LOCAL.context = self.previous


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@dataclass
class LedgerScope:
    """Identity attached to every request recorded by one operation."""

    operation_id: str | None = None
    queue_id: str | None = None
    queue_item_id: str | None = None
    run_id: str | None = None
    provider: str | None = None
    model: str | None = None
    extra: dict = field(default_factory=dict)


def request_identity(
    purpose: str, provider: str, model: str, attempt: int, context: dict
) -> str:
    """Stable request id: identical work reuses one id so retries/resume never double count."""
    return fingerprint({
        "purpose": purpose,
        "provider": provider,
        "model": model,
        "attempt": attempt,
        "task": context.get("task"),
        "record_id": context.get("record_id"),
        "sample_index": context.get("sample_index", 0),
        "config": context.get("config"),
    })[:24]


def make_entry(
    *,
    purpose: str,
    provider: str,
    model: str,
    usage: NormalizedUsage,
    raw_usage: dict | None,
    status: str,
    attempt: int,
    attempts_total: int | None = None,
    error: str | None = None,
    scope: LedgerScope | None = None,
    context: dict | None = None,
    inherited_from: str | None = None,
    sdk_retries_observable: bool = False,
) -> dict:
    context = {**(context or {}), **(scope.extra if scope else {})}
    scope = scope or LedgerScope()
    entry = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "request_id": request_identity(purpose, provider, model, attempt, context),
        "recorded_at": utc_now(),
        "operation_id": scope.operation_id,
        "queue_id": scope.queue_id,
        "queue_item_id": scope.queue_item_id,
        "run_id": scope.run_id,
        "run_directory": context.get("run_directory"),
        "task": context.get("task"),
        "record_id": context.get("record_id"),
        "sample_index": context.get("sample_index", 0),
        "dataset": context.get("dataset"),
        "config_fingerprint": context.get("config"),
        "purpose": purpose,
        "provider": provider,
        "model": model,
        "attempt": attempt,
        "attempts_total": attempts_total,
        "response_status": status,
        "error": error,
        "usage": usage.to_dict(),
        "raw_usage": raw_usage if isinstance(raw_usage, dict) else None,
        "usage_origin": "inherited" if inherited_from else "measured_here",
        "inherited_from": inherited_from,
        "sdk_retries_observable": sdk_retries_observable,
        "counted_total_tokens": usage.counted_total,
    }
    return entry


class UsageLedger:
    """Append-only, fsynced request/attempt ledger with duplicate suppression."""

    def __init__(self, run_dir: Path, scope: LedgerScope | None = None):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / LEDGER_RELATIVE_PATH
        self.scope = scope or LedgerScope()
        self._seen: set[str] = set(self._existing_ids())

    def _existing_ids(self) -> list[str]:
        if not self.path.exists():
            return []
        ids = []
        for entry in iter_ledger(self.path):
            ids.append(entry.get("request_id", ""))
        return [value for value in ids if value]

    def record(self, entry: dict) -> dict | None:
        """Persist one request/attempt. Reused identities are ignored, not duplicated."""
        request_id = entry.get("request_id")
        if request_id and request_id in self._seen:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if request_id:
            self._seen.add(request_id)
        return entry

    def entries(self) -> list[dict]:
        return list(iter_ledger(self.path))

    def write_summary(self, extra: dict | None = None) -> dict:
        summary = summarize_usage(self.entries())
        if extra:
            summary.update(extra)
        atomic_json(self.path.parent / "summary.json", summary)
        return summary


def iter_ledger(path: Path) -> Iterable[dict]:
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn final line after an abrupt stop
    return entries


def ledger_path(run_dir: Path) -> Path:
    return Path(run_dir) / LEDGER_RELATIVE_PATH


def _sum(values: Iterable[int | None]) -> int:
    return sum(value for value in values if value is not None)


def _model_key(entry: dict) -> str:
    return str(entry.get("model") or entry.get("provider") or "unknown")


def _record_key(entry: dict) -> str:
    return f"{entry.get('task')}/{entry.get('record_id')}/{entry.get('sample_index', 0)}"


def _totals(entries: list[dict]) -> dict:
    """Flat measured-token totals plus coverage. Unknown usage is never zero-filled."""
    if not entries:
        return {
            "requests_total": 0,
            "requests_with_known_usage": 0,
            "requests_with_unknown_usage": 0,
            "known_coverage": None,
            "counted_input_tokens": 0,
            "counted_output_tokens": 0,
            "counted_total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }
    measured = [entry for entry in entries if (entry.get("usage") or {}).get("measured")]
    return {
        "requests_total": len(entries),
        "requests_with_known_usage": len(measured),
        "requests_with_unknown_usage": len(entries) - len(measured),
        "known_coverage": len(measured) / len(entries),
        "counted_input_tokens": _sum((entry.get("usage") or {}).get("counted_input_tokens") for entry in entries),
        "counted_output_tokens": _sum((entry.get("usage") or {}).get("counted_output_tokens") for entry in entries),
        "counted_total_tokens": _sum(entry.get("counted_total_tokens") for entry in entries),
        "cache_read_tokens": _sum((entry.get("usage") or {}).get("cache_read_tokens") for entry in entries),
        "cache_write_tokens": _sum((entry.get("usage") or {}).get("cache_write_tokens") for entry in entries),
        "reasoning_tokens": _sum((entry.get("usage") or {}).get("reasoning_tokens") for entry in entries),
    }


def _group(entries: list[dict], key) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for entry in entries:
        buckets.setdefault(str(key(entry)), []).append(entry)
    return {name: _totals(buckets[name]) for name in sorted(buckets)}


def _matches(entry: dict, purpose=None, status=None, model=None, task=None) -> bool:
    return (
        (purpose is None or entry.get("purpose") == purpose)
        and (status is None or entry.get("response_status") == status)
        and (model is None or _model_key(entry) == model)
        and (task is None or str(entry.get("task")) == task)
    )


def summarize_usage(entries: list[dict], since: str | None = None) -> dict:
    """Aggregate a ledger. ``since`` reports usage newly incurred after a resume."""
    unique: dict[str, dict] = {}
    for entry in entries:
        key = entry.get("request_id") or fingerprint(entry)
        unique.setdefault(key, entry)
    deduplicated = list(unique.values())
    fresh = [entry for entry in deduplicated if since is None or str(entry.get("recorded_at")) > since]
    totals = _totals(deduplicated)
    summary = {
        **totals,
        "schema_version": LEDGER_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "entries_read": len(entries),
        "duplicate_entries_suppressed": len(entries) - len(deduplicated),
        "usage_totals_are_exact": totals["requests_with_unknown_usage"] == 0,
        "known_subtotal_only": totals["requests_with_unknown_usage"] > 0,
        "inherited_requests": sum(
            1 for entry in deduplicated if entry.get("usage_origin") == "inherited"
        ),
        "by_purpose": _group(deduplicated, lambda entry: entry.get("purpose")),
        "by_model": _group(deduplicated, _model_key),
        "by_task": _group(deduplicated, lambda entry: entry.get("task")),
        "by_record": _group(deduplicated, _record_key),
        "by_provider": _group(deduplicated, lambda entry: entry.get("provider")),
        "successful_generation": _totals([
            entry for entry in deduplicated
            if _matches(entry, PURPOSE_GENERATION, STATUS_SUCCESS)
        ]),
        "retry_and_failure_overhead": _totals([
            entry for entry in deduplicated
            if not _matches(entry, PURPOSE_GENERATION, STATUS_SUCCESS)
        ]),
        "token_semantics": TOKEN_SEMANTICS,
    }
    if since is not None:
        summary["since"] = since
        summary["new_since_resume"] = _totals(fresh)
    return summary


def summarize_operations(entries: list[dict]) -> dict:
    """Per operation/run/item grouping used by queue and run summaries."""
    return {
        "by_operation": _group(entries, lambda entry: entry.get("operation_id")),
        "by_run": _group(entries, lambda entry: entry.get("run_directory")),
        "by_queue_item": _group(entries, lambda entry: entry.get("queue_item_id")),
    }


def collect_run_ledgers(root: Path) -> list[dict]:
    """Read every run ledger beneath ``root`` in a stable order."""
    root = Path(root)
    entries: list[dict] = []
    for path in sorted(root.rglob(LEDGER_RELATIVE_PATH)):
        entries.extend(iter_ledger(path))
    return entries


def summarize_paths(paths: Iterable[Path], since: str | None = None) -> dict:
    entries: list[dict] = []
    for path in sorted({Path(value) for value in paths}):
        entries.extend(iter_ledger(path))
    return summarize_usage(entries, since=since)


def write_ledger_csv(path: Path, entries: list[dict]) -> Path:
    """Export one tidy CSV row per unique request."""
    import csv

    path = Path(path)
    rows = ledger_rows(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in LEDGER_CSV_COLUMNS})
    return path


def write_ledger_exports(ledger: UsageLedger, extra: dict | None = None) -> dict:
    """Write the run's ledger summary and tidy request CSV next to the ledger."""
    entries = ledger.entries()
    summary = summarize_usage(entries)
    if extra:
        summary.update(extra)
    atomic_json(ledger.path.parent / "summary.json", summary)
    write_ledger_csv(ledger.path.parent / "requests.csv", entries)
    return summary


def ledger_rows(entries: list[dict]) -> list[dict]:
    """Flat per-request rows for CSV export."""
    unique: dict[str, dict] = {}
    for entry in entries:
        key = entry.get("request_id") or fingerprint(entry)
        unique.setdefault(key, entry)
    rows = []
    for key in sorted(unique):
        entry = unique[key]
        usage = entry.get("usage") or {}
        rows.append({
            "request_id": key,
            "recorded_at": entry.get("recorded_at"),
            "purpose": entry.get("purpose"),
            "provider": entry.get("provider"),
            "model": entry.get("model"),
            "task": entry.get("task"),
            "record_id": entry.get("record_id"),
            "sample_index": entry.get("sample_index"),
            "attempt": entry.get("attempt"),
            "attempts_total": entry.get("attempts_total"),
            "response_status": entry.get("response_status"),
            "error_present": bool(entry.get("error")),
            "input_tokens": usage.get("counted_input_tokens"),
            "output_tokens": usage.get("counted_output_tokens"),
            "total_tokens": entry.get("counted_total_tokens"),
            "usage_known": bool(usage.get("measured")),
            "usage_source": usage.get("usage_source"),
            "cache_read_tokens": usage.get("cache_read_tokens"),
            "cache_write_tokens": usage.get("cache_write_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "cache_accounting": usage.get("cache_accounting"),
            "reasoning_accounting": usage.get("reasoning_accounting"),
            "latency_ms": usage.get("latency_ms"),
            "usage_origin": entry.get("usage_origin"),
            "run_directory": entry.get("run_directory"),
            "queue_id": entry.get("queue_id"),
            "queue_item_id": entry.get("queue_item_id"),
        })
    return rows


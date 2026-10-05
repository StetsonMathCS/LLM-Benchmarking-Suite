"""Response-first, resumable FACETS experiment orchestration."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from core.base import (
    BaseProvider,
    BenchmarkResult,
    BenchmarkStatus,
    LLMResponse,
    ModelConfig,
)
from core.registry import ProviderRegistry
from core.scoring import ScoringEngine
from datasets.mapper import DatasetMapper
from facets import EVALUATOR_VERSION, SCHEMA_VERSION
from facets.config import ALL_TASKS, fingerprint, nonsecret_config
from facets.evaluation.execution import ExecutionSettings, configure_default
from facets.runstore import RunStore, atomic_json
from facets.scoring_profile import PROJECT_ROOT, load_profile
from facets.usage import (
    LEDGER_RELATIVE_PATH,
    PURPOSE_EMBEDDING,
    PURPOSE_GENERATION,
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_TRUNCATED,
    LedgerScope,
    UsageLedger,
    get_context,
    make_entry,
    normalize_saved_response,
    normalize_usage,
    UsageContext,
    write_ledger_exports,
)
from utils.code_runner import extract_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_metadata() -> dict:
    def command(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip()
    return {
        "sha": command("rev-parse", "HEAD") or None,
        "dirty": bool(command("status", "--porcelain")),
    }


def _tool_versions() -> dict:
    versions = {}
    for package in ("pytest", "pyyaml", "openai", "anthropic", "ollama", "pylint", "bandit"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    for executable in ("docker", "podman", "node", "eslint", "g++", "cppcheck"):
        path = shutil.which(executable)
        if not path:
            versions[executable] = None
            continue
        result = subprocess.run([path, "--version"], capture_output=True, text=True, check=False, timeout=5)
        versions[executable] = (result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr).strip() else None
    return versions


class ReplayProvider(BaseProvider):
    """Returns one already-persisted response; never performs inference."""

    def __init__(self, config: ModelConfig, response: LLMResponse):
        super().__init__(config)
        self.response = response
        self.calls = 0

    def connect(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("replay response consumed more than once")
        return self.response


def response_from_dict(value: dict) -> LLMResponse:
    allowed = {
        "content", "model", "provider", "prompt_tokens", "completion_tokens", "latency_ms",
        "error", "status", "stop_reason", "attempts", "truncated", "requested_settings",
        "effective_settings", "usage",
    }
    return LLMResponse(**{key: value[key] for key in allowed if key in value})


def _is_transient(error: str) -> bool:
    text = error.lower()
    return any(token in text for token in (
        "timeout", "timed out", "rate limit", "429", "500", "502", "503", "504",
        "temporarily unavailable", "connection reset", "connection error",
    ))


def complete_with_backoff(
    provider: BaseProvider,
    prompt: str,
    config: dict,
    observer=None,
) -> LLMResponse:
    retries = config.get("retries", {}) or {}
    attempts = max(1, int(retries.get("attempts", 3)))
    delay = float(retries.get("initial_backoff_s", 1))
    maximum = float(retries.get("max_backoff_s", 8))
    response = None
    for attempt in range(1, attempts + 1):
        response = provider.complete(prompt, provider.config.system_prompt)
        response.attempts = attempt
        if observer is not None:
            observer(attempt, response, attempts)
        if not response.error or not _is_transient(response.error) or attempt == attempts:
            return response
        time.sleep(min(delay, maximum))
        delay = min(maximum, delay * 2)
    assert response is not None
    return response


def build_model_config(config: dict, model_spec: dict) -> ModelConfig:
    credential_env = model_spec.get("credential_env")
    extra = dict(config.get("provider_extras", {}) or {})
    extra["_provider_timeout_s"] = float((config.get("timeouts", {}) or {}).get("provider_s", 180))
    if model_spec.get("endpoint"):
        extra["_endpoint"] = model_spec["endpoint"]
    return ModelConfig(
        provider=model_spec["provider"],
        model_name=model_spec["model"],
        api_key=os.environ.get(credential_env) if credential_env else None,
        base_url=model_spec.get("base_url") or config.get("base_url"),
        temperature=config.get("temperature"),
        max_tokens=config.get("max_tokens"),
        system_prompt=config.get("system_prompt"),
        extra_params=extra,
    )


def _execution_settings(config: dict) -> ExecutionSettings:
    execution = config.get("execution", {}) or {}
    timeouts = config.get("timeouts", {}) or {}
    return ExecutionSettings(
        backend=execution.get("backend", "container"),
        timeout_s=float(timeouts.get("evaluator_s", 15)),
        memory_mb=int(execution.get("memory_mb", 256)),
        cpus=float(execution.get("cpus", 1.0)),
        pids_limit=int(execution.get("pids_limit", 64)),
        container_image=execution.get("image", "facets-evaluator:revised-v1"),
    )


def _plan(config: dict, model_spec: dict) -> list[dict]:
    mapper = DatasetMapper()
    tasks = config.get("tasks") or ALL_TASKS
    language = config.get("language", "python")
    limit = config.get("limit")
    samples = int(config.get("samples", 1))
    config_fp = fingerprint(nonsecret_config(config, model_spec))
    planned = []
    for task in tasks:
        records = mapper.load_dataset(task, language, limit=limit)
        for record in records:
            for sample_index in range(samples):
                identity_data = {
                    "task": task, "dataset": f"{task}/{language}", "record_id": record.record_id,
                    "record_hash": record.fingerprint, "sample_index": sample_index,
                    "provider": model_spec["provider"], "model": model_spec["model"], "config": config_fp,
                }
                planned.append({
                    "identity": fingerprint(identity_data),
                    "identity_data": identity_data,
                    "record": record,
                })
    return planned


def _manifest(config: dict, model_spec: dict, plan: list[dict]) -> dict:
    profile = load_profile(str(config.get("profile", "revised-v1")))
    mapper = DatasetMapper()
    dataset_info = {
        task: mapper.get_dataset_info(task, config.get("language", "python"))
        for task in config.get("tasks", ALL_TASKS)
    }
    prompts = {}
    for path in sorted((PROJECT_ROOT / "prompts").glob("*.txt")):
        prompts[path.name] = fingerprint(path.read_text(encoding="utf-8"))
    safe_config = nonsecret_config(config, model_spec)
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "scoring_profile": profile.profile_id,
        "scoring_profile_hash": profile.profile_hash,
        "scoring_profile_definition": profile.raw,
        "config_fingerprint": fingerprint(safe_config),
        "resolved_config": safe_config,
        "model_alias": model_spec.get("alias"),
        "provider": model_spec["provider"],
        "model": model_spec["model"],
        "requested_generation_settings": {
            "temperature": config.get("temperature"), "max_tokens": config.get("max_tokens"),
            "provider_extras": config.get("provider_extras", {}),
        },
        "git": _git_metadata(),
        "dataset": dataset_info,
        "prompt_hashes": prompts,
        "dependency_versions": _tool_versions(),
        "runtime": {
            "python": os.sys.version,
            "platform": os.uname().sysname + " " + os.uname().release if hasattr(os, "uname") else os.name,
        },
        "planned_count": len(plan),
        "completed_count": 0,
        "generated_count": 0,
        "reused_response_count": 0,
        "started_at": utc_now(),
        "completed_at": None,
        "status": "running",
    }


def _prepare_item(item: dict, config: dict, model_config: ModelConfig) -> tuple[object, dict, str, type, str]:
    record = item["record"]
    kwargs = DatasetMapper().map_record_to_benchmark_kwargs(record)
    code_input = kwargs.pop("code_input", "")
    if item["identity_data"]["task"] == "translation":
        kwargs["target_language"] = config.get("translation_target", "javascript")
    benchmark_cls = DatasetMapper().get_task_class(item["identity_data"]["task"])
    placeholder = ReplayProvider(model_config, LLMResponse("", model_config.model_name, model_config.provider))
    prompt_builder = benchmark_cls(config.get("language", "python"), placeholder)
    prompt = prompt_builder.build_prompt(config.get("language", "python"), code_input, **kwargs)
    return record, kwargs, code_input, benchmark_cls, prompt


def _usage_block(summary: dict) -> dict:
    """Compact, report-embedded usage view; full detail stays in usage/summary.json."""
    return {
        "requests_total": summary["requests_total"],
        "requests_with_known_usage": summary["requests_with_known_usage"],
        "requests_with_unknown_usage": summary["requests_with_unknown_usage"],
        "known_coverage": summary["known_coverage"],
        "counted_input_tokens": summary["counted_input_tokens"],
        "counted_output_tokens": summary["counted_output_tokens"],
        "counted_total_tokens": summary["counted_total_tokens"],
        "generation": _usage_scope(summary["by_purpose"], PURPOSE_GENERATION),
        "embedding": _usage_scope(summary["by_purpose"], PURPOSE_EMBEDDING),
        "inherited_requests": summary["inherited_requests"],
        "totals_are_exact": summary["usage_totals_are_exact"],
        "detail": "usage/summary.json",
    }


def _usage_scope(by_purpose: dict, purpose: str) -> dict:
    scope = (by_purpose or {}).get(purpose) or {}
    return {
        "requests_total": scope.get("requests_total", 0),
        "requests_with_unknown_usage": scope.get("requests_with_unknown_usage", 0),
        "counted_input_tokens": scope.get("counted_input_tokens", 0),
        "counted_output_tokens": scope.get("counted_output_tokens", 0),
        "counted_total_tokens": scope.get("counted_total_tokens", 0),
    }


def _item_context(item: dict, config: dict, manifest: dict) -> dict:
    """Request identity fields for one planned record."""
    identity = item["identity_data"]
    return {
        "task": identity.get("task"),
        "record_id": identity.get("record_id"),
        "sample_index": identity.get("sample_index", 0),
        "dataset": identity.get("dataset"),
        "config": manifest.get("config_fingerprint"),
        "provider": identity.get("provider"),
        "model": identity.get("model"),
    }


class UsageRecorder:
    """Records every provider attempt, embedding call, and reused response.

    Only provider-reported counts are counted. Reused responses are recorded as
    inherited so an operation's totals include work paid for earlier while
    keeping that distinction visible.
    """

    def __init__(self, ledger: UsageLedger, attempts_total: int):
        self.ledger = ledger
        self.attempts_total = attempts_total

    def observe_generation(self, attempt: int, response: LLMResponse, attempts_total: int | None) -> None:
        usage = normalize_usage(response.provider, response.usage or None)
        if response.error:
            status = STATUS_FAILED
        elif response.truncated:
            status = STATUS_TRUNCATED
        else:
            status = STATUS_SUCCESS
        self.ledger.record(make_entry(
            purpose=PURPOSE_GENERATION,
            provider=response.provider,
            model=response.model,
            usage=usage,
            raw_usage=response.usage,
            status=status,
            attempt=attempt,
            attempts_total=attempts_total or self.attempts_total,
            error=response.error,
            scope=self.ledger.scope,
            context=get_context(),
            sdk_retries_observable=True,
        ))

    def observe_inherited(self, response: LLMResponse) -> None:
        usage = normalize_saved_response(response.provider, response.to_dict())
        self.ledger.record(make_entry(
            purpose=PURPOSE_GENERATION,
            provider=response.provider,
            model=response.model,
            usage=usage,
            raw_usage=response.usage,
            status=STATUS_SUCCESS if response.success else STATUS_FAILED,
            attempt=max(1, int(response.attempts or 1)),
            attempts_total=self.attempts_total,
            error=response.error,
            scope=self.ledger.scope,
            context=get_context(),
            inherited_from="saved_response",
            sdk_retries_observable=True,
        ))

    def generation_observer(self, context: dict | None = None) -> Callable[..., None]:
        """Observer bound to one record's context, safe to call from any thread."""
        def observe(attempt: int, response: LLMResponse, attempts_total: int | None) -> None:
            if context is None:
                self.observe_generation(attempt, response, attempts_total)
                return
            with UsageContext(**context):
                self.observe_generation(attempt, response, attempts_total)
        return observe

    def embedding_sink(self) -> Callable[[dict], None]:
        """Sink for embedding dimensions: billable embedding work, accounted apart."""
        def sink(payload: dict) -> None:
            self.record_embedding(
                provider=payload.get("provider") or "ollama",
                model=payload.get("model"),
                raw_usage=payload.get("raw_usage"),
                latency_ms=payload.get("latency_ms"),
                status=payload.get("status", STATUS_FAILED),
                error=payload.get("error"),
            )
        return sink

    def record_embedding(
        self,
        provider: str,
        model: str,
        raw_usage: dict | None,
        latency_ms: float | None,
        status: str,
        error: str | None = None,
    ) -> None:
        usage = normalize_usage(provider, raw_usage)
        self.ledger.record(make_entry(
            purpose=PURPOSE_EMBEDDING,
            provider=provider,
            model=model,
            usage=usage.with_latency(latency_ms),
            raw_usage=raw_usage,
            status=status,
            attempt=1,
            scope=self.ledger.scope,
            context=get_context(),
            sdk_retries_observable=True,
        ))


def execute_run(
    config: dict,
    model_spec: dict,
    run_dir: Path,
    *,
    resume: bool = False,
    provider_override: BaseProvider | None = None,
    usage_scope: LedgerScope | None = None,
    record_usage: bool = True,
) -> tuple[Path, int]:
    profile = load_profile(str(config.get("profile", "revised-v1")))
    configure_default(_execution_settings(config))
    embedding = config.get("embedding", {}) or {}
    similarity_dimension = None
    if embedding or record_usage:
        from benchmarks.dimensions.code_review import (
            ReferenceReviewSimilarityDimension as SimilarityDimension,
        )
        similarity_dimension = SimilarityDimension
    if embedding:
        similarity_dimension.DEFAULT_BASE_URL = embedding.get("base_url", similarity_dimension.DEFAULT_BASE_URL)
        similarity_dimension.EMBEDDING_MODEL = embedding.get("model", similarity_dimension.EMBEDDING_MODEL)
    plan = _plan(config, model_spec)
    store = RunStore(run_dir)
    manifest = _manifest(config, model_spec, plan)
    store.initialize(manifest, resume=resume)
    if resume:
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "running"
        manifest["completed_at"] = None
    completed = store.completed_identities()
    model_config = build_model_config(config, model_spec)
    retries = config.get("retries", {}) or {}
    scope = usage_scope or LedgerScope(run_id=str(run_dir))
    scope.extra = {**scope.extra, "run_directory": str(run_dir), "config": manifest["config_fingerprint"]}
    ledger = UsageLedger(run_dir, scope)
    recorder = UsageRecorder(ledger, max(1, int(retries.get("attempts", 3))))
    embedding_sink = recorder.embedding_sink() if record_usage else None
    previous_embedding_sink = None
    if embedding_sink is not None and similarity_dimension is not None:
        previous_embedding_sink = similarity_dimension.USAGE_SINK
        similarity_dimension.USAGE_SINK = embedding_sink
    generation_provider = provider_override
    generated_count = sum(
        1 for path in store.responses.glob("*.json")
        if json.loads(path.read_text(encoding="utf-8")).get("origin") == "generated"
    )
    reused_count = int(manifest.get("reused_response_count", 0))
    interrupted = False
    generated_this_run: set[str] = set()

    try:
        generation_concurrency = max(1, int(config.get("generation_concurrency", 1)))
        if generation_concurrency > 1:
            pending = [item for item in plan if item["identity"] not in completed and store.load_response(item["identity"]) is None]
            if pending and generation_provider is None:
                generation_provider = ProviderRegistry.create(model_config)
            with ThreadPoolExecutor(max_workers=generation_concurrency) as pool:
                futures = {}
                for item in pending:
                    prompt = _prepare_item(item, config, model_config)[4]
                    context = _item_context(item, config, manifest)
                    future = pool.submit(
                        complete_with_backoff,
                        generation_provider,
                        prompt,
                        config,
                        recorder.generation_observer(context),
                    )
                    futures[future] = (item, prompt)
                for future in as_completed(futures):
                    item, prompt = futures[future]
                    response = future.result()
                    identity = item["identity"]
                    store.save_response(identity, {
                        "schema_version": SCHEMA_VERSION, "record_identity": identity, "prompt": prompt,
                        "response": response.to_dict(), "saved_at": utc_now(), "origin": "generated",
                    })
                    generated_this_run.add(identity)
                    generated_count += 1
        for item in plan:
            identity = item["identity"]
            if identity in completed:
                continue
            record, kwargs, code_input, benchmark_cls, prompt = _prepare_item(item, config, model_config)

            saved = store.load_response(identity)
            reused = saved is not None and identity not in generated_this_run
            with UsageContext(**_item_context(item, config, manifest)):
                if saved is None:
                    if generation_provider is None:
                        generation_provider = ProviderRegistry.create(model_config)
                    response = complete_with_backoff(
                        generation_provider, prompt, config, recorder.observe_generation
                    )
                    saved = {
                        "schema_version": SCHEMA_VERSION,
                        "record_identity": identity,
                        "prompt": prompt,
                        "response": response.to_dict(),
                        "saved_at": utc_now(),
                        "origin": "generated",
                    }
                    store.save_response(identity, saved)
                    generated_count += 1
                else:
                    if reused:
                        reused_count += 1
                        recorder.observe_inherited(response_from_dict(saved["response"]))
                response = response_from_dict(saved["response"])
                replay = ReplayProvider(model_config, response)
                benchmark = benchmark_cls(config.get("language", "python"), replay)
                result = benchmark._timed_run(code_input=code_input, **kwargs)
            result.metadata.update(item["identity_data"])
            result.metadata["response_reused"] = reused
            result.metadata["record_identity"] = identity
            result_payload = result.to_dict()
            result_payload.update({
                "schema_version": SCHEMA_VERSION,
                "evaluator_version": EVALUATOR_VERSION,
                "record_identity": identity,
                "task": item["identity_data"]["task"],
                "dataset_id": item["identity_data"]["dataset"],
                "record_id": record.record_id,
                "sample_index": item["identity_data"]["sample_index"],
                "prompt": prompt,
                "raw_response_text": response.content,
                "extracted_candidate": extract_code(response.content),
                "response_status": response.status if not response.error else "provider_error",
                "response_stop_reason": response.stop_reason,
                "attempts": response.attempts,
                "applicable_weights": profile.dimension_weights[item["identity_data"]["task"]],
                "evaluated_at": utc_now(),
            })
            store.append_result(result_payload)
            completed.add(identity)
            store.checkpoint({
                "planned": len(plan), "completed": len(completed), "generated": generated_count,
                "reused_responses": reused_count, "updated_at": utc_now(),
            })
    except KeyboardInterrupt:
        interrupted = True

    results = []
    for value in store.iter_results():
        result = BenchmarkResult(
            benchmark_name=value.get("benchmark", value.get("task", "unknown")),
            status=BenchmarkStatus(value["status"]),
            combined_score=value.get("combined_score"),
            metadata={**value.get("metadata", {}), "task_name": value.get("task")},
        )
        results.append(result)
    report = ScoringEngine(
        results,
        task_weights=profile.task_weights,
        pass_threshold=profile.composite_threshold,
        num_samples=int(config.get("samples", 1)),
        pass_k_values=config.get("pass_k", [1]),
    ).compute()
    report_payload = report.to_dict()
    selected_tasks = list(config.get("tasks", ALL_TASKS))
    selected_weight_total = sum(profile.task_weights[task] for task in selected_tasks)
    report_payload["applicable_task_weights"] = {
        task: profile.task_weights[task] / selected_weight_total for task in selected_tasks
    }
    dimension_values: dict[str, list[float]] = {}
    dimension_statuses: dict[str, dict[str, int]] = {}
    for value in store.iter_results():
        for dimension_id, dimension in (value.get("details") or {}).items():
            if not isinstance(dimension, dict):
                continue
            if dimension.get("score") is not None:
                dimension_values.setdefault(dimension_id, []).append(float(dimension["score"]))
            status = dimension.get("status", "unknown")
            statuses = dimension_statuses.setdefault(dimension_id, {})
            statuses[status] = statuses.get(status, 0) + 1
    report_payload["dimension_scores"] = {
        dimension_id: {
            "mean_score": sum(values) / len(values) if values else None,
            "record_count": len(values),
            "statuses": dimension_statuses.get(dimension_id, {}),
        }
        for dimension_id, values in dimension_values.items()
    }
    usage_summary = write_ledger_exports(ledger)
    report_payload["token_usage"] = _usage_block(usage_summary)
    atomic_json(store.run_dir / "summary.json", report_payload)
    manifest["completed_count"] = len(completed)
    manifest["generated_count"] = generated_count
    manifest["reused_response_count"] = reused_count
    manifest["completed_at"] = utc_now()
    manifest["status"] = "interrupted" if interrupted else ("complete" if len(completed) == len(plan) and report.complete else "incomplete")
    effective_settings = sorted({
        json.dumps((value.get("llm_response") or {}).get("effective_settings", {}), sort_keys=True)
        for value in store.iter_results()
    })
    manifest["effective_generation_settings"] = [json.loads(value) for value in effective_settings]
    manifest["mutant_pool_hashes"] = sorted({
        ((value.get("details") or {}).get("generated_test_effectiveness") or {}).get("details", {}).get("mutant_pool_hash")
        for value in store.iter_results()
        if ((value.get("details") or {}).get("generated_test_effectiveness") or {}).get("details", {}).get("mutant_pool_hash")
    })
    manifest["token_usage"] = _usage_block(usage_summary)
    manifest["usage_ledger"] = str(LEDGER_RELATIVE_PATH)
    atomic_json(store.manifest_path, manifest)
    if similarity_dimension is not None and (previous_embedding_sink is not None or embedding_sink is not None):
        similarity_dimension.USAGE_SINK = previous_embedding_sink
    if interrupted:
        return store.run_dir, 130
    return store.run_dir, 0 if manifest["status"] == "complete" else 3

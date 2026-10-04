"""Response-first, resumable FACETS experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from benchmarks import matrix
from core.base import BaseProvider, BenchmarkResult, BenchmarkStatus, LLMResponse, ModelConfig
from core.registry import ProviderRegistry
from core.scoring import ScoringEngine
from datasets.mapper import DatasetMapper
from facets import EVALUATOR_VERSION, SCHEMA_VERSION
from facets.config import ALL_TASKS, fingerprint, nonsecret_config
from facets.evaluation.execution import ExecutionSettings, configure_default
from facets.runstore import RunStore, atomic_json
from facets.scoring_profile import PROJECT_ROOT, load_profile
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

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("replay response consumed more than once")
        return self.response


def response_from_dict(value: dict) -> LLMResponse:
    allowed = {
        "content", "model", "provider", "prompt_tokens", "completion_tokens", "latency_ms",
        "error", "status", "stop_reason", "attempts", "truncated", "requested_settings", "effective_settings",
    }
    return LLMResponse(**{key: value[key] for key in allowed if key in value})


def _is_transient(error: str) -> bool:
    text = error.lower()
    return any(token in text for token in (
        "timeout", "timed out", "rate limit", "429", "500", "502", "503", "504",
        "temporarily unavailable", "connection reset", "connection error",
    ))


def complete_with_backoff(provider: BaseProvider, prompt: str, config: dict) -> LLMResponse:
    retries = config.get("retries", {}) or {}
    attempts = max(1, int(retries.get("attempts", 3)))
    delay = float(retries.get("initial_backoff_s", 1))
    maximum = float(retries.get("max_backoff_s", 8))
    response = None
    for attempt in range(1, attempts + 1):
        response = provider.complete(prompt, provider.config.system_prompt)
        response.attempts = attempt
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
        "dataset_selection": config.get("selection", {}),
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


def execute_run(
    config: dict,
    model_spec: dict,
    run_dir: Path,
    *,
    resume: bool = False,
    provider_override: BaseProvider | None = None,
) -> tuple[Path, int]:
    profile = load_profile(str(config.get("profile", "revised-v1")))
    configure_default(_execution_settings(config))
    embedding = config.get("embedding", {}) or {}
    if embedding:
        from benchmarks.dimensions.code_review import ReferenceReviewSimilarityDimension
        ReferenceReviewSimilarityDimension.DEFAULT_BASE_URL = embedding.get("base_url", ReferenceReviewSimilarityDimension.DEFAULT_BASE_URL)
        ReferenceReviewSimilarityDimension.EMBEDDING_MODEL = embedding.get("model", ReferenceReviewSimilarityDimension.EMBEDDING_MODEL)
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
                    future = pool.submit(complete_with_backoff, generation_provider, prompt, config)
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
            if saved is None:
                if generation_provider is None:
                    generation_provider = ProviderRegistry.create(model_config)
                response = complete_with_backoff(generation_provider, prompt, config)
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
    atomic_json(store.manifest_path, manifest)
    if interrupted:
        return store.run_dir, 130
    return store.run_dir, 0 if manifest["status"] == "complete" else 3

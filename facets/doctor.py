"""Offline preflight validation; performs no model inference or downloads."""

from __future__ import annotations

import ast
import json
import keyword
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import urllib.parse

from datasets.mapper import DatasetMapper
from facets.config import ALL_TASKS, validate_resolved
from facets.evaluation.execution import ContainerBackend, ExecutionSettings, get_backend
from facets.scoring_profile import load_profile


def run_doctor(config: dict, model_spec: dict, registry: dict, live_provider_check: bool = False) -> dict:
    issues = validate_resolved(config, model_spec, registry)
    mapper = DatasetMapper()
    counts = {}
    hashes = {}
    records_by_task = {}
    language = config.get("language", "python")
    for task in config.get("tasks", ALL_TASKS):
        try:
            records = mapper.load_dataset(task, language)
            records_by_task[task] = records
            info = mapper.get_dataset_info(task, language)
            counts[task] = len(records)
            hashes[task] = info["dataset_hash"]
            if not info["record_ids_unique"]:
                issues.append({"severity": "error", "code": "duplicate_record_id", "detail": task})
            for row in mapper.malformed_rows(task, language):
                issues.append({
                    "severity": "error",
                    "code": "dataset_row_malformed",
                    "detail": (
                        f"{task}:{language} line {row['line']} has {row['actual_fields']} fields, "
                        f"expected {row['expected_fields']} (unescaped comma or quote?)"
                    ),
                })
            for record in records:
                data = record.data
                if task == "code_generation":
                    entry = data.get("entry_point", "")
                    tests = data.get("test", "")
                    if not entry.isidentifier() or keyword.iskeyword(entry):
                        issues.append({"severity": "error", "code": "entry_point", "detail": f"{task}:{record.record_id}"})
                    try:
                        tree = ast.parse(tests)
                        if not any(isinstance(node, ast.FunctionDef) and node.name == "check" for node in tree.body):
                            raise ValueError("missing check")
                    except Exception as exc:
                        issues.append({"severity": "error", "code": "reference_checker", "detail": f"{task}:{record.record_id}: {exc}"})
                if task == "test_generation" and not data.get("test", "").strip():
                    issues.append({"severity": "error", "code": "reference_checker", "detail": f"{task}:{record.record_id}"})
                if task == "refactoring" and (not data.get("test_harness", "").strip() or data.get("expected_console_output") is None):
                    issues.append({"severity": "error", "code": "runtime_workload", "detail": f"{task}:{record.record_id}"})
        except Exception as exc:
            issues.append({"severity": "error", "code": "dataset", "detail": f"{task}: {exc}"})

    expected = config.get("expected_counts") or {}
    for task, expected_count in expected.items():
        if counts.get(task) != expected_count:
            issues.append({"severity": "error", "code": "dataset_count", "detail": f"{task}: expected {expected_count}, found {counts.get(task)}"})

    selected_tasks = set(config.get("tasks", ALL_TASKS))
    if selected_tasks & {"bug_fixing", "refactoring", "translation"}:
        for tool in ("pylint", "bandit"):
            if shutil.which(tool) is None:
                issues.append({"severity": "error", "code": "missing_tool", "detail": tool})
    if "translation" in selected_tasks and config.get("translation_target", "javascript") == "javascript":
        for tool in ("node", "eslint"):
            if shutil.which(tool) is None:
                issues.append({"severity": "error", "code": "missing_tool", "detail": tool})
    execution = config.get("execution", {}) or {}
    backend_ready = True
    if execution.get("backend", "container") == "container":
        backend = ContainerBackend(ExecutionSettings(container_image=execution.get("image", "facets-evaluator:revised-v1")))
        if not backend.available():
            backend_ready = False
            issues.append({"severity": "error", "code": "container_runtime", "detail": "docker or podman not found"})
        else:
            inspect = subprocess.run(
                [backend.runtime, "image", "inspect", execution.get("image", "facets-evaluator:revised-v1")],
                capture_output=True, text=True, check=False,
            )
            if inspect.returncode != 0:
                backend_ready = False
                issues.append({"severity": "error", "code": "container_image", "detail": execution.get("image", "facets-evaluator:revised-v1")})
    elif execution.get("backend") == "local":
        issues.append({"severity": "warning", "code": "local_backend", "detail": "local subprocess is not a filesystem/network sandbox"})
    else:
        backend_ready = False
        issues.append({"severity": "error", "code": "execution_backend", "detail": execution.get("backend")})

    if config.get("mutation_preflight", "full") == "full" and backend_ready and "test_generation" in records_by_task:
        from benchmarks.dimensions.generated_test_effectiveness import build_mutants
        from benchmarks.dimensions.reference_test_success import ReferenceTestSuccessDimension
        timeouts = config.get("timeouts", {}) or {}
        settings = ExecutionSettings(
            backend=execution.get("backend", "container"),
            timeout_s=float(timeouts.get("evaluator_s", 15)),
            memory_mb=int(execution.get("memory_mb", 256)),
            cpus=float(execution.get("cpus", 1.0)),
            pids_limit=int(execution.get("pids_limit", 64)),
            container_image=execution.get("image", "facets-evaluator:revised-v1"),
        )
        reference = ReferenceTestSuccessDimension(get_backend(settings), settings)
        for record in records_by_task["test_generation"]:
            source = record.data.get("code_snippet", "")
            entry = record.data.get("entry_point", "")
            trusted = record.data.get("test", "")
            baseline = reference.evaluate("python", source, test=trusted, entry_point=entry)
            if baseline.status != "ok":
                issues.append({"severity": "error", "code": "gte_baseline", "detail": record.record_id})
                continue
            eligible = 0
            for mutant in build_mutants(source, entry):
                result = reference.evaluate("python", mutant["source"], test=trusted, entry_point=entry)
                eligible += int(result.status == "candidate_failure")
            if eligible == 0:
                issues.append({"severity": "error", "code": "gte_mutant_pool", "detail": record.record_id})

    if "code_review" in config.get("tasks", ALL_TASKS):
        embedding_url = config.get("embedding", {}).get("base_url", "http://localhost:11434")
        try:
            with urllib.request.urlopen(embedding_url.rstrip("/") + "/api/tags", timeout=2) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
        except Exception as exc:
            issues.append({"severity": "error", "code": "embedding_service", "detail": str(exc)})
    if live_provider_check:
        provider = model_spec.get("provider")
        model = urllib.parse.quote(str(model_spec.get("model", "")), safe="")
        try:
            if provider == "openai":
                request = urllib.request.Request(
                    f"https://api.openai.com/v1/models/{model}",
                    headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
                )
            elif provider == "anthropic":
                request = urllib.request.Request(
                    f"https://api.anthropic.com/v1/models/{model}",
                    headers={"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01"},
                )
            elif provider == "ollama":
                base = model_spec.get("base_url", "http://localhost:11434").rstrip("/")
                request = urllib.request.Request(base + "/api/tags")
            else:
                raise RuntimeError(f"no capability adapter for provider {provider}")
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if provider == "ollama":
                installed = {item.get("name") or item.get("model") for item in payload.get("models", [])}
                if model_spec.get("model") not in installed:
                    raise RuntimeError(f"exact local tag not installed: {model_spec.get('model')}")
        except Exception as exc:
            issues.append({"severity": "error", "code": "live_provider_capability", "detail": str(exc)})
    issues.append({"severity": "warning", "code": "dataset_provenance", "detail": "per-record source/license fields are incomplete in current CSV schemas"})
    return {
        "ok": not any(issue["severity"] == "error" for issue in issues),
        "profile": load_profile(str(config.get("profile", "revised-v1"))).profile_id,
        "dataset_counts": counts,
        "dataset_hashes": hashes,
        "planned_records_per_model": sum(counts.values()) * int(config.get("samples", 1)),
        "issues": issues,
        "provider_inference_performed": False,
        "downloads_performed": False,
    }

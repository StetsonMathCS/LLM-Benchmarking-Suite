"""Noninteractive command-line interface for FACETS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from core.base import BaseProvider, LLMResponse, ModelConfig
from facets.analysis import analyze_directory, summarize_directory
from facets.config import (
    ALL_TASKS, fingerprint, load_model_registry, load_yaml, merge_cli, nonsecret_config,
    resolve_model, validate_resolved,
)
from facets.doctor import run_doctor
from facets.migration import migrate_legacy_report, parse_legacy_repr
from facets.runner import _manifest, _plan, execute_run
from facets.runstore import RunStore, atomic_json


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _provider_extras(values: list[str] | None) -> dict | None:
    if values is None:
        return None
    extras = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"provider extra must be KEY=JSON_VALUE: {value!r}")
        key, raw = value.split("=", 1)
        try:
            extras[key] = json.loads(raw)
        except json.JSONDecodeError:
            extras[key] = raw
    return extras


def _load_resolved(args) -> tuple[dict, dict, dict]:
    config = load_yaml(args.config)
    overrides = {
        "tasks": getattr(args, "tasks", None),
        "language": getattr(args, "language", None),
        "translation_target": getattr(args, "translation_target", None),
        "limit": getattr(args, "limit", None),
        "samples": getattr(args, "samples", None),
        "temperature": getattr(args, "temperature", None),
        "max_tokens": getattr(args, "max_tokens", None),
        "output_dir": getattr(args, "output_dir", None),
        "run_name": getattr(args, "run_name", None),
        "generation_concurrency": getattr(args, "generation_concurrency", None),
        "evaluator_concurrency": getattr(args, "evaluator_concurrency", None),
        "provider_extras": _provider_extras(getattr(args, "provider_extra", None)),
    }
    config = merge_cli(config, overrides)
    registry = load_model_registry(getattr(args, "registry", None))
    model_spec = resolve_model(
        getattr(args, "alias", None), getattr(args, "provider", None), getattr(args, "model", None), registry
    )
    return config, model_spec, registry


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("alias", nargs="?")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS)
    parser.add_argument("--language")
    parser.add_argument("--translation-target")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--provider-extra", action="append", metavar="KEY=VALUE")
    parser.add_argument("--provider-timeout", type=float)
    parser.add_argument("--evaluator-timeout", type=float)
    parser.add_argument("--retries", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--generation-concurrency", type=int)
    parser.add_argument("--evaluator-concurrency", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facets", description="FACETS reproducible software-engineering benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    models = sub.add_parser("models", help="model registry commands")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    list_parser = models_sub.add_parser("list")
    list_parser.add_argument("--registry")

    doctor = sub.add_parser("doctor", help="offline configuration and evaluator preflight")
    doctor.add_argument("--config", required=True)
    doctor.add_argument("--registry")
    doctor.add_argument("--live-provider-check", action="store_true")

    run = sub.add_parser("run", help="run one model noninteractively")
    _add_run_options(run)

    sweep = sub.add_parser("sweep", help="run every model listed in the config")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--registry")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--resume", action="store_true")
    sweep.add_argument("--limit", type=int)
    sweep.add_argument("--output-dir")

    resume = sub.add_parser("resume", help="resume an interrupted run directory")
    resume.add_argument("run_directory")

    reevaluate = sub.add_parser("reevaluate", help="reevaluate saved responses without generation calls")
    reevaluate.add_argument("source")
    reevaluate.add_argument("--profile", default="revised-v1")
    reevaluate.add_argument("--output", required=True)
    reevaluate.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "experiments" / "final-study.yaml"))

    summarize = sub.add_parser("summarize", help="summarize structured runs")
    summarize.add_argument("results_directory")
    summarize.add_argument("--output")

    analyze = sub.add_parser("analyze", help="export tables, sensitivity, bootstrap intervals, and charts")
    analyze.add_argument("results_directory")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--allow-mixed", action="store_true")
    return parser


def _run_command(args) -> int:
    config, model_spec, registry = _load_resolved(args)
    if args.provider_timeout is not None or args.evaluator_timeout is not None:
        timeouts = dict(config.get("timeouts", {}) or {})
        if args.provider_timeout is not None:
            timeouts["provider_s"] = args.provider_timeout
        if args.evaluator_timeout is not None:
            timeouts["evaluator_s"] = args.evaluator_timeout
        config["timeouts"] = timeouts
    if args.retries is not None:
        retry = dict(config.get("retries", {}) or {})
        retry["attempts"] = args.retries
        config["retries"] = retry
    issues = validate_resolved(config, model_spec, registry)
    _print({"resolved_nonsecret_config": nonsecret_config(config, model_spec), "preflight_issues": issues})
    if args.dry_run:
        doctor = run_doctor(config, model_spec, registry)
        _print(doctor)
        return 0 if doctor["ok"] else 2
    if any(issue["severity"] == "error" for issue in issues):
        return 2
    output_root = Path(config.get("output_dir", "reports/runs"))
    run_name = config.get("run_name") or model_spec.get("alias") or model_spec["model"].replace("/", "_")
    path, status = execute_run(config, model_spec, output_root / run_name)
    print(path)
    return status


def _sweep_command(args) -> int:
    config = merge_cli(load_yaml(args.config), {"limit": args.limit, "output_dir": args.output_dir})
    registry = load_model_registry(args.registry)
    statuses = []
    for model_index, alias in enumerate(config.get("models", [])):
        model_spec = resolve_model(alias, None, None, registry)
        issues = validate_resolved(config, model_spec, registry)
        if args.dry_run:
            doctor_config = config if model_index == 0 else {**config, "mutation_preflight": "already_validated_for_shared_dataset"}
            statuses.append({"alias": alias, "doctor": run_doctor(doctor_config, model_spec, registry)})
            continue
        if any(issue["severity"] == "error" for issue in issues):
            statuses.append({"alias": alias, "status": "preflight_failed", "issues": issues})
            continue
        output = Path(config.get("output_dir", "reports/runs")) / alias
        path, code = execute_run(config, model_spec, output, resume=args.resume)
        statuses.append({"alias": alias, "path": str(path), "exit_code": code})
    _print(statuses)
    return 0 if all(item.get("exit_code", 0) == 0 and item.get("doctor", {"ok": True}).get("ok", True) for item in statuses) else 3


class _NoGenerationProvider(BaseProvider):
    def connect(self): return True
    def is_available(self): return False
    def complete(self, prompt, system_prompt=None):
        raise RuntimeError("reevaluate attempted a forbidden generation call")


def _reevaluate(args) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    config = load_yaml(args.config)
    config["profile"] = args.profile
    if source.is_dir() and (source / "manifest.json").exists():
        source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        source_results = {}
        with (source / "results.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                source_results[(record["task"], str(record["record_id"]), int(record.get("sample_index", 0)))] = record
        model_spec = {"alias": source_manifest.get("model_alias"), "provider": source_manifest["provider"], "model": source_manifest["model"], "credential_env": None, "availability": "saved_response"}
        source_provenance = source_manifest
    else:
        legacy = json.loads(source.read_text(encoding="utf-8"))
        model_cfg = legacy.get("model_config", {})
        model_spec = {"alias": None, "provider": model_cfg.get("provider", "legacy"), "model": model_cfg.get("model_name", "legacy"), "credential_env": None, "availability": "saved_response"}
        counters, source_results = {}, {}
        task_by_benchmark = {"Bug Fixing Benchmark": "bug_fixing", "Code Generation Benchmark": "code_generation", "Code Review Benchmark": "code_review", "Refactoring Benchmark": "refactoring", "Test Generation Benchmark": "test_generation", "Translation Benchmark": "translation"}
        for legacy_record in legacy.get("results", []):
            task = task_by_benchmark.get(legacy_record.get("benchmark"))
            if not task:
                continue
            index = counters.get(task, 0)
            counters[task] = index + 1
            try:
                response = parse_legacy_repr(legacy_record.get("llm_response", ""))
            except Exception:
                response = None
            source_results[(task, str(index + 1), 0)] = {"raw_response_text": response.get("content", "") if response else None}
        source_provenance = legacy
    plan = _plan(config, model_spec)
    store = RunStore(output)
    manifest = _manifest(config, model_spec, plan)
    manifest["reevaluation_source"] = str(source)
    manifest["reevaluation_source_hash"] = fingerprint(source_provenance)
    store.initialize(manifest)
    if not source.is_dir():
        migrate_legacy_report(source, output / "legacy-adapter")
    missing = []
    for item in plan:
        key = (item["identity_data"]["task"], str(item["identity_data"]["record_id"]), item["identity_data"]["sample_index"])
        old = source_results.get(key)
        content = old.get("raw_response_text") if old else None
        if content is None:
            missing.append(key)
            continue
        response = LLMResponse(content=content, model=model_spec["model"], provider=model_spec["provider"], status="saved_response")
        store.save_response(item["identity"], {"schema_version": "2.0", "record_identity": item["identity"], "prompt": None, "response": response.to_dict(), "reevaluation_source": str(source), "origin": "reevaluation_source"})
    if missing:
        atomic_json(output / "reevaluation-error.json", {"missing_saved_responses": missing})
        print(f"reevaluation refused: {len(missing)} responses could not be recovered", file=sys.stderr)
        return 4
    _, code = execute_run(config, model_spec, output, resume=True, provider_override=_NoGenerationProvider(ModelConfig(model_spec["provider"], model_spec["model"])))
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "models":
            _print(load_model_registry(args.registry).get("models", {}))
            return 0
        if args.command == "doctor":
            config = load_yaml(args.config)
            registry = load_model_registry(args.registry)
            reports = []
            for model_index, alias in enumerate(config.get("models", [])):
                doctor_config = config if model_index == 0 else {**config, "mutation_preflight": "already_validated_for_shared_dataset"}
                reports.append({"alias": alias, **run_doctor(doctor_config, resolve_model(alias, None, None, registry), registry, args.live_provider_check)})
            _print(reports)
            return 0 if reports and all(report["ok"] for report in reports) else 2
        if args.command == "run": return _run_command(args)
        if args.command == "sweep": return _sweep_command(args)
        if args.command == "resume":
            run_dir = Path(args.run_directory).resolve()
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            config = dict(manifest["resolved_config"])
            model_spec = config.pop("model")
            return execute_run(config, model_spec, run_dir, resume=True)[1]
        if args.command == "reevaluate": return _reevaluate(args)
        if args.command == "summarize":
            rows = summarize_directory(Path(args.results_directory))
            _print(rows)
            if args.output: atomic_json(Path(args.output), {"runs": rows})
            return 0
        if args.command == "analyze":
            _print(analyze_directory(Path(args.results_directory), Path(args.output), args.allow_mixed))
            return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"facets: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

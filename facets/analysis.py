"""Structured, manifest-compatible FACETS result summaries and analysis."""

from __future__ import annotations

import csv
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from facets.figures import figure_manifest, render_figures, write_index
from facets.runstore import atomic_json
from facets.scoring_profile import load_profile

# Directories that hold queue bookkeeping, checkpoints, or previous analysis
# output. They may contain manifests but are not experiment runs.
NON_RUN_DIRECTORIES = {"queues", "checkpoints", "analysis", "figures"}


def discover_runs(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        return [root]
    return sorted(
        path.parent
        for path in root.rglob("manifest.json")
        if not NON_RUN_DIRECTORIES.intersection(path.relative_to(root).parts)
    )


def select_runs(
    runs: list[Path],
    root: Path,
    selection: str | None = None,
    run_ids: list[str] | None = None,
    allow_mixed: bool = False,
) -> list[Path]:
    """Resolve a run set deterministically, refusing ambiguous repeats.

    ``selection`` is ``None`` (every model must appear once), ``"latest-complete"``
    (newest complete run per model) or ``"explicit"`` (exactly the given run IDs).
    """
    if not runs:
        raise ValueError(f"no structured runs found under {root}")
    by_id = {run_id(run, root): run for run in runs}
    if selection == "explicit" or run_ids:
        wanted = list(dict.fromkeys(str(run_id).strip("/") for run_id in run_ids or []))
        missing = [run_id for run_id in wanted if run_id not in by_id]
        if missing:
            raise ValueError(f"requested run ids not found: {missing}")
        return [by_id[run_id] for run_id in wanted]
    if selection not in (None, "latest-complete"):
        raise ValueError(f"unknown run selection: {selection}")

    by_model: dict[str, list[Path]] = {}
    manifests = {}
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        manifests[run] = manifest
        by_model.setdefault(_model_name(manifest), []).append(run)
    repeated = {model: runs for model, runs in by_model.items() if len(runs) > 1}
    if repeated and selection is None and not allow_mixed:
        detail = {model: [run_id(run, root) for run in runs] for model, runs in sorted(repeated.items())}
        raise ValueError(f"ambiguous repeated models {detail}; pass --run <id>... or --latest-complete")
    chosen = []
    for model, model_runs in sorted(by_model.items()):
        if selection == "latest-complete" and len(model_runs) > 1:
            complete = [run for run in model_runs if manifests[run].get("status") == "complete"] or model_runs
            chosen.append(max(complete, key=lambda run: _completion_key(run, manifests[run])))
        else:
            chosen.extend(model_runs)
    return chosen


def run_id(run: Path, root: Path) -> str:
    """Stable identifier for a run, relative to the analysed root when possible."""
    try:
        return str(run.relative_to(root))
    except ValueError:
        return str(run)


def _completion_key(run: Path, manifest: dict) -> tuple[str, float]:
    stamp = str(manifest.get("completed_at") or manifest.get("generated_at") or "")
    return stamp, run.stat().st_mtime


def load_run(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    records = []
    results_path = path / "results.jsonl"
    if results_path.exists():
        with results_path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    return manifest, records


def ensure_compatible(manifests: list[dict], allow_mixed: bool = False) -> list[str]:
    incomplete = [manifest.get("model_alias") or manifest.get("model") for manifest in manifests if manifest.get("status") != "complete"]
    if incomplete and not allow_mixed:
        raise ValueError(f"incomplete runs cannot enter official comparison: {incomplete}")
    signatures = set()
    for manifest in manifests:
        dataset_hashes = tuple(sorted(
            (task, info.get("dataset_hash")) for task, info in manifest.get("dataset", {}).items()
        ))
        signatures.add((manifest.get("evaluator_version"), manifest.get("scoring_profile_hash"), dataset_hashes))
    if len(signatures) > 1 and not allow_mixed:
        raise ValueError("incompatible evaluator/profile/dataset manifests; use an explicit mixed descriptive analysis")
    warnings = ["mixed-run descriptive comparison; inferential/rank results suppressed"] if len(signatures) > 1 else []
    if incomplete:
        warnings.append(f"incomplete-run descriptive comparison: {incomplete}")
    return warnings


def _model_name(manifest: dict) -> str:
    return manifest.get("model_alias") or manifest.get("model") or "unknown"


def summarize_directory(root: Path) -> list[dict]:
    rows = []
    for run_dir in discover_runs(root):
        manifest, records = load_run(run_dir)
        valid = [record["combined_score"] for record in records if record.get("combined_score") is not None]
        composite_passes = [score for score in valid if score >= 0.5]
        rows.append({
            "run_directory": str(run_dir),
            "model": _model_name(manifest),
            "status": manifest.get("status"),
            "evaluator_version": manifest.get("evaluator_version"),
            "profile": manifest.get("scoring_profile"),
            "planned": manifest.get("planned_count"),
            "completed": len(records),
            "mean_composite_score": mean(valid) if valid else None,
            "composite_pass_rate": len(composite_passes) / len(records) if records else None,
            "infrastructure_failures": sum(record.get("combined_score") is None for record in records),
        })
    return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or _insertion_order(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _insertion_order(rows: list[dict]) -> list[str]:
    """Columns in first-seen order so tables read predictably, not alphabetically."""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _rank(values: dict[str, float]) -> dict[str, int]:
    return {name: index + 1 for index, (name, _) in enumerate(sorted(values.items(), key=lambda item: (-item[1], item[0])))}


def _spearman(a: dict[str, int], b: dict[str, int]) -> float | None:
    names = sorted(set(a) & set(b))
    if len(names) < 2:
        return None
    squared = sum((a[name] - b[name]) ** 2 for name in names)
    return 1 - (6 * squared) / (len(names) * (len(names) ** 2 - 1))


def _aggregate(records: list[dict], task_weights: dict[str, float], equal_dimensions: bool = False) -> float | None:
    by_task: dict[str, list[float]] = {}
    for record in records:
        if record.get("combined_score") is None:
            return None
        score = record["combined_score"]
        if equal_dimensions:
            dimensions = [
                value.get("score") for value in (record.get("details") or {}).values()
                if isinstance(value, dict) and value.get("applicable", True) and value.get("score") is not None
            ]
            score = mean(dimensions) if dimensions else score
        by_task.setdefault(record["task"], []).append(score)
    active = {task: mean(values) for task, values in by_task.items() if values}
    denominator = sum(task_weights.get(task, 0) for task in active)
    return sum(task_weights.get(task, 0) * score for task, score in active.items()) / denominator if denominator else None


def _bootstrap_pair(records_a: list[dict], records_b: list[dict], weights: dict[str, float], seed: int, replicates: int = 2000) -> dict:
    keyed_a = {(record["task"], record["record_id"], record.get("sample_index", 0)): record for record in records_a}
    keyed_b = {(record["task"], record["record_id"], record.get("sample_index", 0)): record for record in records_b}
    common = sorted(set(keyed_a) & set(keyed_b))
    strata: dict[str, list[float]] = {}
    for key in common:
        a = keyed_a[key].get("combined_score")
        b = keyed_b[key].get("combined_score")
        if a is not None and b is not None:
            strata.setdefault(key[0], []).append(a - b)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        value = 0.0
        denominator = sum(weights.get(task, 0) for task, values in strata.items() if values)
        for task, values in sorted(strata.items()):
            sampled = [values[rng.randrange(len(values))] for _ in values]
            value += (weights.get(task, 0) / denominator) * mean(sampled)
        draws.append(value)
    draws.sort()
    return {
        "paired_records": sum(len(values) for values in strata.values()),
        "mean_difference": mean(draws) if draws else None,
        "ci_95_low": draws[int(0.025 * len(draws))] if draws else None,
        "ci_95_high": draws[min(len(draws) - 1, int(0.975 * len(draws)))] if draws else None,
        "seed": seed,
        "replicates": replicates,
        "interpretation": "variation across paired benchmark records, not repeated-inference uncertainty or guaranteed generalization",
    }


def analyze_directory(
    root: Path,
    output: Path,
    allow_mixed: bool = False,
    selection: str | None = None,
    run_ids: list[str] | None = None,
) -> dict:
    discovered = discover_runs(root)
    runs = select_runs(discovered, root, selection=selection, run_ids=run_ids, allow_mixed=allow_mixed)
    loaded = [(path, *load_run(path)) for path in runs]
    warnings = ensure_compatible([item[1] for item in loaded], allow_mixed=allow_mixed)
    skipped = [str(path) for path in discovered if path not in set(runs)]
    if skipped:
        warnings.append(f"runs excluded by selection={selection or 'default'}: {skipped}")
    output.mkdir(parents=True, exist_ok=True)
    tidy_records = []
    tidy_dimensions = []
    runtime_rows = []
    latency_rows = []
    model_records = {}
    manifests = {}
    for run_dir, manifest, records in loaded:
        model = _model_name(manifest)
        manifests[model] = manifest
        model_records[model] = records
        for record in records:
            metadata = record.get("metadata") or {}
            response = record.get("llm_response") or {}
            usage = response.get("usage") or {}
            latency_ms = response.get("latency_ms")
            tidy_records.append({
                "model": model, "task": record.get("task"), "record_id": record.get("record_id"),
                "sample_index": record.get("sample_index", 0), "combined_score": record.get("combined_score"),
                "composite_pass": record.get("combined_score") is not None and record["combined_score"] >= 0.5,
                "functional_correct": metadata.get("functional_correct"), "status": record.get("status"),
                "record_identity": record.get("record_identity"),
                "latency_ms": latency_ms,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            })
            if latency_ms is not None:
                latency_rows.append({"model": model, "task": record.get("task"),
                                     "record_id": record.get("record_id"), "latency_ms": latency_ms})
            for dimension_id, result in (record.get("details") or {}).items():
                if not isinstance(result, dict):
                    continue
                tidy_dimensions.append({
                    "model": model, "task": record.get("task"), "record_id": record.get("record_id"),
                    "sample_index": record.get("sample_index", 0), "dimension_id": dimension_id,
                    "display_name": result.get("display_name"), "score": result.get("score"),
                    "status": result.get("status"), "applicable": result.get("applicable"),
                    "functional_correct": metadata.get("functional_correct"),
                })
                if dimension_id == "runtime_analysis":
                    detail = result.get("details") or {}
                    runtime_rows.append({
                        "model": model, "record_id": record.get("record_id"),
                        "functional_correct": metadata.get("functional_correct"),
                        "raw_time_ratio": detail.get("raw_time_ratio"),
                        "raw_memory_ratio": detail.get("raw_memory_ratio"),
                    })
    _write_csv(output / "records.csv", tidy_records)
    _write_csv(output / "dimensions.csv", tidy_dimensions)
    _write_csv(output / "runtime_ratios.csv", runtime_rows)
    _write_csv(output / "generation_latency.csv", latency_rows)

    task_summary = []
    dimension_summary = []
    for model in sorted(model_records):
        for task in sorted({row["task"] for row in tidy_records if row["model"] == model}):
            values = [row["combined_score"] for row in tidy_records if row["model"] == model and row["task"] == task and row["combined_score"] is not None]
            functional = [row for row in tidy_records if row["model"] == model and row["task"] == task and row["functional_correct"] is not None]
            task_summary.append({
                "model": model, "task": task, "records": len(values), "mean_score": mean(values) if values else None,
                "functional_pass_rate": mean([float(row["functional_correct"]) for row in functional]) if functional else None,
            })
        for dimension in sorted({row["dimension_id"] for row in tidy_dimensions if row["model"] == model}):
            rows = [row for row in tidy_dimensions if row["model"] == model and row["dimension_id"] == dimension and row["score"] is not None]
            correct = [row["score"] for row in rows if row["functional_correct"] is True and dimension not in {"functional_correctness", "reference_test_success"}]
            dimension_summary.append({
                "model": model, "dimension_id": dimension, "records": len(rows),
                "mean_score": mean([row["score"] for row in rows]) if rows else None,
                "mean_among_functionally_correct": mean(correct) if correct else None,
            })
    _write_csv(output / "task_summary.csv", task_summary)
    _write_csv(output / "dimension_summary.csv", dimension_summary)

    profile = load_profile(next(iter(manifests.values())).get("scoring_profile", "revised-v1")) if manifests else load_profile()
    perturbations = {
        "frozen_default": profile.task_weights,
        "equal_task_weights": {task: 1 / len(profile.task_weights) for task in profile.task_weights},
        "generation_plus_20pct": {task: weight * (1.2 if task == "code_generation" else 1.0) for task, weight in profile.task_weights.items()},
        "review_plus_20pct": {task: weight * (1.2 if task == "code_review" else 1.0) for task, weight in profile.task_weights.items()},
    }
    perturbations = {name: {task: value / sum(weights.values()) for task, value in weights.items()} for name, weights in perturbations.items()}
    scores_by_profile = {}
    for name, weights in perturbations.items():
        equal_dimensions = name == "equal_dimensions"
        scores_by_profile[name] = {
            model: score for model, records in model_records.items()
            if (score := _aggregate(records, weights, equal_dimensions=equal_dimensions)) is not None
        }
    scores_by_profile["equal_dimensions"] = {
        model: score for model, records in model_records.items()
        if (score := _aggregate(records, profile.task_weights, equal_dimensions=True)) is not None
    }
    default_rank = _rank(scores_by_profile.get("frozen_default", {}))
    sensitivity = []
    for name, scores in scores_by_profile.items():
        ranks = _rank(scores)
        for model, score in scores.items():
            sensitivity.append({
                "profile": name, "model": model, "score": score, "rank": ranks[model],
                "rank_change_from_default": default_rank.get(model, ranks[model]) - ranks[model],
                "spearman_with_default": _spearman(default_rank, ranks),
            })
    _write_csv(output / "sensitivity.csv", sensitivity)

    bootstrap = []
    models = sorted(model_records)
    if not warnings:
        for index, model_a in enumerate(models):
            for model_b in models[index + 1:]:
                bootstrap.append({"model_a": model_a, "model_b": model_b, **_bootstrap_pair(
                    model_records[model_a], model_records[model_b], profile.task_weights,
                    seed=20260404 + index * 100 + models.index(model_b),
                )})
    _write_csv(output / "paired_bootstrap.csv", bootstrap)

    default_scores = scores_by_profile.get("frozen_default", {})
    known_parameter_sizes = {}
    undisclosed_parameter_sizes = []
    for model, manifest in manifests.items():
        raw_size = ((manifest.get("resolved_config") or {}).get("model") or {}).get("parameter_size")
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([BM])\s*", str(raw_size or ""), re.IGNORECASE)
        if match:
            value = float(match.group(1)) * (1.0 if match.group(2).upper() == "B" else 0.001)
            known_parameter_sizes[model] = value
        else:
            undisclosed_parameter_sizes.append(model)
    _write_csv(output / "parameter_sizes.csv", [
        {"model": model, "parameter_size_billions": known_parameter_sizes.get(model), "status": "known" if model in known_parameter_sizes else "undisclosed_or_unverified"}
        for model in sorted(manifests)
    ])

    tasks = sorted({row["task"] for row in task_summary if row["task"]})
    figures = render_figures(
        output=output,
        records=tidy_records,
        task_summary=task_summary,
        model_scores=default_scores,
        parameter_sizes=known_parameter_sizes,
        undisclosed_sizes=sorted(undisclosed_parameter_sizes),
        latency_rows=latency_rows,
        models=models,
        tasks=tasks,
    )
    warnings.extend(f"figure unavailable: {failure}" for failure in figures["failures"])
    index = write_index(output, figures["figures"], {
        "generated_at": _generated_at(),
        "runs": len(loaded),
        "evaluator_version": next(iter(manifests.values())).get("evaluator_version") if manifests else None,
        "scoring_profile": next(iter(manifests.values())).get("scoring_profile") if manifests else None,
        "models": models,
        "warnings": warnings,
    })
    figure_manifest(figures, output, index)

    summary = {
        "models": models,
        "runs": len(loaded),
        "run_ids": [run_id(path, root) for path in runs],
        "selection": selection or ("explicit" if run_ids else "default"),
        "warnings": warnings,
        "chart_status": "generated" if figures["figures"] else "not generated",
        "figures": figures["stems"],
        "figure_directory": output.name,
        "known_parameter_sizes_billions": known_parameter_sizes,
        "undisclosed_or_unverified_parameter_sizes": sorted(undisclosed_parameter_sizes),
        "correctness_note": "functional correctness uses actual applicable functional outcomes; composite pass is reported separately",
        "runtime_note": "raw time/memory ratios are exported separately and only correctness-gated measurements should be interpreted",
        "latency_note": "provider-reported generation latency only; hardware and provider paths differ between models",
        "cross_benchmark_note": "no numeric FACETS/LiveCodeBench equivalence is computed",
    }
    atomic_json(output / "analysis.json", summary)
    return summary


def _generated_at() -> str:
    return datetime.now(UTC).isoformat()

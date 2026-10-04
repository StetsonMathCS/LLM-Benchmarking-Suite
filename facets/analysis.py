"""Structured, manifest-compatible FACETS result summaries and analysis."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
import random
from statistics import mean
from typing import Iterable

from facets.runstore import atomic_json
from facets.scoring_profile import load_profile


def discover_runs(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("manifest.json"))


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
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def analyze_directory(root: Path, output: Path, allow_mixed: bool = False) -> dict:
    loaded = [(path, *load_run(path)) for path in discover_runs(root)]
    warnings = ensure_compatible([item[1] for item in loaded], allow_mixed=allow_mixed)
    output.mkdir(parents=True, exist_ok=True)
    tidy_records = []
    tidy_dimensions = []
    runtime_rows = []
    model_records = {}
    manifests = {}
    for run_dir, manifest, records in loaded:
        model = _model_name(manifest)
        manifests[model] = manifest
        model_records[model] = records
        for record in records:
            metadata = record.get("metadata") or {}
            tidy_records.append({
                "model": model, "task": record.get("task"), "record_id": record.get("record_id"),
                "sample_index": record.get("sample_index", 0), "combined_score": record.get("combined_score"),
                "composite_pass": record.get("combined_score") is not None and record["combined_score"] >= 0.5,
                "functional_correct": metadata.get("functional_correct"), "status": record.get("status"),
                "record_identity": record.get("record_identity"),
            })
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

    chart_status = "not generated"
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
    try:
        import matplotlib.pyplot as plt
        default_scores = scores_by_profile.get("frozen_default", {})
        if default_scores:
            names = sorted(default_scores, key=default_scores.get, reverse=True)
            fig, axis = plt.subplots(figsize=(10, 5))
            axis.bar(names, [default_scores[name] for name in names])
            axis.set_ylabel("FACETS composite score")
            axis.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(output / "model_composite_scores.png", dpi=300)
            plt.close(fig)
            tasks = sorted({row["task"] for row in task_summary})
            fig, axes = plt.subplots(len(tasks), 1, figsize=(10, max(4, 2.5 * len(tasks))), squeeze=False)
            for axis, task in zip(axes[:, 0], tasks):
                rows = sorted((row for row in task_summary if row["task"] == task), key=lambda row: row["model"])
                axis.bar([row["model"] for row in rows], [row["mean_score"] for row in rows])
                axis.set_title(task)
                axis.set_ylim(0, 1)
                axis.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(output / "task_comparisons.png", dpi=300)
            plt.close(fig)
            dimensions = sorted({row["dimension_id"] for row in dimension_summary})
            fig, axes = plt.subplots(len(dimensions), 1, figsize=(10, max(4, 2.5 * len(dimensions))), squeeze=False)
            for axis, dimension in zip(axes[:, 0], dimensions):
                rows = sorted((row for row in dimension_summary if row["dimension_id"] == dimension), key=lambda row: row["model"])
                axis.bar([row["model"] for row in rows], [row["mean_score"] for row in rows])
                axis.set_title(dimension)
                axis.set_ylim(0, 1)
                axis.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(output / "dimension_comparisons.png", dpi=300)
            plt.close(fig)
            sized = [(model, known_parameter_sizes[model], default_scores[model]) for model in known_parameter_sizes if model in default_scores]
            if sized:
                fig, axis = plt.subplots(figsize=(8, 5))
                axis.scatter([row[1] for row in sized], [row[2] for row in sized])
                for model, size, score in sized:
                    axis.annotate(model, (size, score))
                axis.set_xlabel("Known parameter size (billions)")
                axis.set_ylabel("FACETS composite score")
                fig.tight_layout()
                fig.savefig(output / "known_parameter_size_relationship.png", dpi=300)
                plt.close(fig)
            measured_runtime = [row for row in runtime_rows if row["functional_correct"] is True and row["raw_time_ratio"] is not None and row["raw_memory_ratio"] is not None]
            if measured_runtime:
                fig, axis = plt.subplots(figsize=(8, 5))
                for model in sorted({row["model"] for row in measured_runtime}):
                    rows = [row for row in measured_runtime if row["model"] == model]
                    axis.scatter([row["raw_time_ratio"] for row in rows], [row["raw_memory_ratio"] for row in rows], label=model, alpha=0.7)
                axis.axvline(1, color="grey", linewidth=0.8)
                axis.axhline(1, color="grey", linewidth=0.8)
                axis.set_xlabel("Baseline/generated time ratio (uncapped)")
                axis.set_ylabel("Baseline/generated peak-memory ratio (uncapped)")
                axis.legend(fontsize="small")
                fig.tight_layout()
                fig.savefig(output / "correct_runtime_raw_ratios.png", dpi=300)
                plt.close(fig)
            chart_status = "generated"
    except Exception as exc:
        warnings.append(f"charts unavailable: {exc}")
    summary = {
        "models": models,
        "runs": len(loaded),
        "warnings": warnings,
        "chart_status": chart_status,
        "known_parameter_sizes_billions": known_parameter_sizes,
        "undisclosed_or_unverified_parameter_sizes": sorted(undisclosed_parameter_sizes),
        "correctness_note": "functional correctness uses actual applicable functional outcomes; composite pass is reported separately",
        "runtime_note": "raw time/memory ratios are exported separately and only correctness-gated measurements should be interpreted",
        "cross_benchmark_note": "no numeric FACETS/LiveCodeBench equivalence is computed",
    }
    atomic_json(output / "analysis.json", summary)
    return summary

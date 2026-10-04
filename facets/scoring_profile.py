"""Validated, versioned FACETS scoring profiles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LEGACY_DIMENSION_NAMES = {
    "Code Generation Tests": "reference_test_success",
    "Code Completion Tests": "reference_test_success",
    "Test Pass Rate": "generated_test_effectiveness",
    "Semantic Drift": "structural_similarity",
    "Code Review Quality": "reference_review_similarity",
    "Code Consistency": "code_consistency",
    "Functional Correctness": "functional_correctness",
    "Linting": "linting",
    "Runtime Analysis": "runtime_analysis",
    "Vulnerabilities": "vulnerabilities",
}


@dataclass(frozen=True)
class ScoringProfile:
    profile_id: str
    evaluator_version: str
    composite_threshold: float
    task_weights: dict[str, float]
    dimension_weights: dict[str, dict[str, float]]
    not_applicable: dict[str, list[str]]
    raw: dict[str, Any]
    profile_hash: str


def _require_unit_sum(values: dict[str, float], label: str) -> None:
    if not values:
        raise ValueError(f"{label} may not be empty")
    if any(not isinstance(value, (int, float)) or value < 0 or value > 1 for value in values.values()):
        raise ValueError(f"{label} weights must each be in [0, 1]")
    total = sum(values.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{label} weights must sum to 1.0 exactly; got {total:.12g}")


def load_profile(name_or_path: str = "revised-v1") -> ScoringProfile:
    path = Path(name_or_path)
    if not path.suffix:
        path = PROJECT_ROOT / "profiles" / f"{name_or_path}.yaml"
    elif not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"scoring profile not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    task_weights = {str(k): float(v) for k, v in raw.get("task_weights", {}).items()}
    dimension_weights = {
        str(task): {str(k): float(v) for k, v in weights.items()}
        for task, weights in raw.get("dimension_weights", {}).items()
    }
    _require_unit_sum(task_weights, "task")
    if set(task_weights) != set(dimension_weights):
        raise ValueError("task_weights and dimension_weights must declare the same six tasks")
    for task, weights in dimension_weights.items():
        _require_unit_sum(weights, f"{task} dimension")
    threshold = float(raw.get("composite_threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError("composite_threshold must be in [0, 1]")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return ScoringProfile(
        profile_id=str(raw["profile_id"]),
        evaluator_version=str(raw["evaluator_version"]),
        composite_threshold=threshold,
        task_weights=task_weights,
        dimension_weights=dimension_weights,
        not_applicable={str(k): list(v) for k, v in raw.get("not_applicable", {}).items()},
        raw=raw,
        profile_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


DEFAULT_PROFILE = load_profile("revised-v1")


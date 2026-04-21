"""
core/scoring.py

Scoring and grading engine for LLM benchmark results.

Aggregation hierarchy:
  dimension scores  (per record, per dimension — produced by each BaseDimension)
      ↓  weighted sum (DIMENSION_WEIGHTS in matrix.py)
  combined_score    (per record — stored on BenchmarkResult)
      ↓  mean across records
  task_score        (per task — one float per benchmark type)
      ↓  weighted sum (TASK_WEIGHTS below)
  final_score       (overall LLM score — single float in [0, 1])
      ↓  GRADE_THRESHOLDS
  grade             (letter grade A+ … F)

Usage:
    engine = ScoringEngine(results)
    report = engine.compute()
    print(report.final_score, report.grade)
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

from core.base import BenchmarkResult, BenchmarkStatus

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum combined_score for a single record to count as "passed"
PASS_THRESHOLD: float = 0.5

# Per-task weights used to compute the final LLM score.
# Weights are re-normalised over only the tasks that were actually run,
# so partial runs still produce a valid 0–1 score.
# Rationale:
#   - bug_fixing / code_generation / code_review / refactoring carry more
#     weight because they probe deeper reasoning.
#   - test_generation is important but more mechanical.
#   - translation is a narrower skill.
TASK_WEIGHTS: dict[str, float] = {
    "bug_fixing":      0.15,
    "code_generation": 0.27,
    "code_review":     0.15,
    "refactoring":     0.15,
    "test_generation": 0.12,
    "translation":     0.16,
}
# Verify at import time that weights sum to ~1.0
assert abs(sum(TASK_WEIGHTS.values()) - 1.0) < 1e-9, "TASK_WEIGHTS must sum to 1.0"

# Category groupings — used for sub-scores that reveal WHERE a model is strong/weak
TASK_CATEGORIES: dict[str, list[str]] = {
    "generation":     ["code_generation", "test_generation"],
    "transformation": ["bug_fixing", "refactoring", "translation"],
    "analysis":       ["code_review"],
}

# (min_score_inclusive, letter_grade) — checked top-down
GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (0.93, "A+"),
    (0.87, "A"),
    (0.80, "A-"),
    (0.73, "B+"),
    (0.67, "B"),
    (0.60, "B-"),
    (0.53, "C+"),
    (0.47, "C"),
    (0.40, "C-"),
    (0.33, "D+"),
    (0.27, "D"),
    (0.20, "D-"),
    (0.00, "F"),
]


# ---------------------------------------------------------------------------
# pass@k estimator  (Chen et al., 2021 — "Evaluating Large Language Models
# Trained on Code")
# ---------------------------------------------------------------------------

def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k.

    Args:
        n: total number of samples generated for a single problem
        c: number of correct samples (combined_score >= threshold)
        k: the k in pass@k

    Returns:
        Estimated probability that at least one of k random samples is correct.
        Returns float('nan') when n < k (insufficient samples).
    """
    if n < k:
        return float("nan")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskScore:
    """Aggregated statistics for one benchmark task."""
    task_name: str
    record_count: int       # total records attempted
    scored_count: int       # records with a valid combined_score
    error_count: int        # records where LLM/evaluation fully failed
    mean_score: float       # mean combined_score over scored records
    pass_rate: float        # fraction of records that scored >= PASS_THRESHOLD
    min_score: float
    max_score: float
    std_score: float        # 0.0 if only one record
    pass_at_k: dict[int, float] = field(default_factory=dict)  # {k: estimated_pass_at_k}

    def to_dict(self) -> dict:
        d = asdict(self)
        # Only include pass_at_k when populated (multi-sample mode)
        if not self.pass_at_k:
            d.pop("pass_at_k", None)
        return d


@dataclass
class BenchmarkReport:
    """
    Complete scoring report for one test run.
    Produced by ScoringEngine.compute().
    """
    # Per-task breakdown
    task_scores: dict[str, TaskScore]

    # Category sub-scores (simple mean of task mean_scores within the category)
    category_scores: dict[str, float]

    # Overall LLM score and grade
    final_score: float      # 0.0 – 1.0, weighted across tasks
    grade: str              # A+, A, A-, B+, …, F

    # Aggregate record counts
    total_records: int
    total_scored: int       # records with valid combined_score
    total_passed: int       # records with combined_score >= PASS_THRESHOLD
    total_errors: int       # records with no combined_score (LLM/critical error)
    overall_pass_rate: float

    # pass@k per task — populated only in multi-sample mode
    # { task_name: {k: estimated_pass_at_k} }
    pass_at_k: dict[str, dict[int, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "final_score": self.final_score,
            "grade": self.grade,
            "overall_pass_rate": self.overall_pass_rate,
            "total_records": self.total_records,
            "total_scored": self.total_scored,
            "total_passed": self.total_passed,
            "total_errors": self.total_errors,
            "category_scores": self.category_scores,
            "task_scores": {k: v.to_dict() for k, v in self.task_scores.items()},
        }
        if self.pass_at_k:
            d["pass_at_k"] = self.pass_at_k
        return d


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ScoringEngine:
    """
    Aggregates a list of BenchmarkResult objects into a BenchmarkReport.

    Results must have ``metadata["task_name"]`` set (done by TestSuite.run_all).
    Falls back to ``result.benchmark_name`` if the key is absent.

    Pass ``task_weights`` / ``pass_threshold`` to override the module-level
    defaults — the TUI uses this to apply user-edited values.

    For multi-sample (pass@k) mode, set ``num_samples`` > 1 and provide
    ``pass_k_values`` (e.g. [1, 5, 10]).  Results must carry
    ``metadata["record_id"]`` so the engine can group samples per problem.
    """

    def __init__(
        self,
        results: list[BenchmarkResult],
        task_weights: Optional[dict[str, float]] = None,
        pass_threshold: Optional[float] = None,
        num_samples: int = 1,
        pass_k_values: Optional[list[int]] = None,
    ):
        self._results = results
        self._weights = task_weights if task_weights is not None else TASK_WEIGHTS
        self._threshold = pass_threshold if pass_threshold is not None else PASS_THRESHOLD
        self._num_samples = num_samples
        self._pass_k_values = pass_k_values or [1, 5, 10]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def compute(self) -> BenchmarkReport:
        """Run the full scoring pipeline and return a BenchmarkReport."""
        task_scores = self._compute_task_scores()
        category_scores = self._compute_category_scores(task_scores)
        final = self._compute_final_score(task_scores)
        grade = self._assign_grade(final)

        total = len(self._results)
        scored = [r for r in self._results if r.combined_score is not None]
        passed = [r for r in scored if r.combined_score >= self._threshold]
        errors = [r for r in self._results if r.combined_score is None]

        # Compute pass@k when in multi-sample mode
        pak = self._compute_pass_at_k(task_scores)

        return BenchmarkReport(
            task_scores=task_scores,
            category_scores=category_scores,
            final_score=final,
            grade=grade,
            total_records=total,
            total_scored=len(scored),
            total_passed=len(passed),
            total_errors=len(errors),
            overall_pass_rate=round(len(passed) / total, 4) if total else 0.0,
            pass_at_k=pak,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _group_by_task(self) -> dict[str, list[BenchmarkResult]]:
        """Group results by task key (e.g. 'bug_fixing')."""
        groups: dict[str, list[BenchmarkResult]] = {}
        for r in self._results:
            key = r.metadata.get("task_name") or r.benchmark_name
            groups.setdefault(key, []).append(r)
        return groups

    def _compute_task_scores(self) -> dict[str, TaskScore]:
        task_scores: dict[str, TaskScore] = {}
        for task_name, records in self._group_by_task().items():
            scored_vals = [
                r.combined_score for r in records if r.combined_score is not None
            ]
            errors = sum(1 for r in records if r.combined_score is None)

            if not scored_vals:
                task_scores[task_name] = TaskScore(
                    task_name=task_name,
                    record_count=len(records),
                    scored_count=0,
                    error_count=errors,
                    mean_score=0.0,
                    pass_rate=0.0,
                    min_score=0.0,
                    max_score=0.0,
                    std_score=0.0,
                )
                continue

            mean = statistics.mean(scored_vals)
            n_passed = sum(1 for s in scored_vals if s >= self._threshold)

            task_scores[task_name] = TaskScore(
                task_name=task_name,
                record_count=len(records),
                scored_count=len(scored_vals),
                error_count=errors,
                mean_score=round(mean, 4),
                pass_rate=round(n_passed / len(records), 4),
                min_score=round(min(scored_vals), 4),
                max_score=round(max(scored_vals), 4),
                std_score=round(
                    statistics.stdev(scored_vals) if len(scored_vals) > 1 else 0.0, 4
                ),
            )
        return task_scores

    def _compute_pass_at_k(
        self, task_scores: dict[str, TaskScore]
    ) -> dict[str, dict[int, float]]:
        """
        Compute pass@k for each task using the unbiased estimator.

        Groups results by (task_name, record_id), counts correct samples per
        problem, applies ``pass_at_k_estimator`` for each requested k, then
        averages across problems.  Also attaches per-task pass_at_k to the
        corresponding TaskScore objects.

        Returns an empty dict when num_samples <= 1 (single-run mode).
        """
        if self._num_samples <= 1:
            return {}

        # Group results by (task, record_id)
        groups: dict[str, dict[str, list[BenchmarkResult]]] = {}
        for r in self._results:
            task = r.metadata.get("task_name") or r.benchmark_name
            rid = r.metadata.get("record_id", "unknown")
            groups.setdefault(task, {}).setdefault(rid, []).append(r)

        pak: dict[str, dict[int, float]] = {}
        for task_name, records_by_id in groups.items():
            k_sums: dict[int, float] = {k: 0.0 for k in self._pass_k_values}
            k_counts: dict[int, int] = {k: 0 for k in self._pass_k_values}

            for _rid, samples in records_by_id.items():
                n = len(samples)
                c = sum(
                    1
                    for s in samples
                    if s.combined_score is not None
                    and s.combined_score >= self._threshold
                )
                for k in self._pass_k_values:
                    val = pass_at_k_estimator(n, c, k)
                    if not math.isnan(val):
                        k_sums[k] += val
                        k_counts[k] += 1

            task_pak: dict[int, float] = {}
            for k in self._pass_k_values:
                if k_counts[k] > 0:
                    task_pak[k] = round(k_sums[k] / k_counts[k], 4)
            pak[task_name] = task_pak

            # Attach to TaskScore object as well
            if task_name in task_scores:
                task_scores[task_name].pass_at_k = task_pak

        return pak

    def _compute_category_scores(
        self, task_scores: dict[str, TaskScore]
    ) -> dict[str, float]:
        """Simple mean of task mean_scores within each category."""
        category_scores: dict[str, float] = {}
        for category, tasks in TASK_CATEGORIES.items():
            scores = [
                task_scores[t].mean_score
                for t in tasks
                if t in task_scores and task_scores[t].scored_count > 0
            ]
            category_scores[category] = round(statistics.mean(scores), 4) if scores else 0.0
        return category_scores

    def _compute_final_score(self, task_scores: dict[str, TaskScore]) -> float:
        """
        Weighted average of per-task mean_scores.

        Only tasks that have at least one scored record contribute.
        Their weights are re-normalised so the result stays in [0, 1]
        even when only a subset of tasks was run.
        """
        active = {
            t: ts for t, ts in task_scores.items() if ts.scored_count > 0
        }
        if not active:
            return 0.0

        total_weight = sum(self._weights.get(t, 0.0) for t in active)

        if total_weight == 0.0:
            return round(statistics.mean(ts.mean_score for ts in active.values()), 4)

        weighted_sum = sum(
            self._weights.get(t, 0.0) * ts.mean_score for t, ts in active.items()
        )
        return round(weighted_sum / total_weight, 4)

    @staticmethod
    def _assign_grade(score: float) -> str:
        for threshold, letter in GRADE_THRESHOLDS:
            if score >= threshold:
                return letter
        return "F"

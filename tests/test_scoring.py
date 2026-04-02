"""
End-to-end tests for the ScoringEngine.
"""
import pytest
from core.scoring import (
    ScoringEngine,
    BenchmarkReport,
    TaskScore,
    PASS_THRESHOLD,
    TASK_WEIGHTS,
    GRADE_THRESHOLDS,
)
from core.base import BenchmarkResult, BenchmarkStatus


def _make_result(task_name: str, score: float, status=BenchmarkStatus.PASSED) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_name=task_name,
        status=status,
        combined_score=score,
        metadata={"task_name": task_name},
    )


def _make_error_result(task_name: str) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_name=task_name,
        status=BenchmarkStatus.ERROR,
        combined_score=None,
        metadata={"task_name": task_name},
    )


# ── Grade assignment ──────────────────────────────────────────────────────

class TestGradeAssignment:
    def test_perfect_score(self):
        assert ScoringEngine._assign_grade(1.0) == "A+"

    def test_a_grade(self):
        assert ScoringEngine._assign_grade(0.90) == "A"

    def test_b_grade(self):
        assert ScoringEngine._assign_grade(0.73) == "B+"
        assert ScoringEngine._assign_grade(0.70) == "B"

    def test_f_grade(self):
        assert ScoringEngine._assign_grade(0.0) == "F"

    def test_boundary_value(self):
        assert ScoringEngine._assign_grade(0.93) == "A+"
        assert ScoringEngine._assign_grade(0.929) == "A"


# ── Single task scoring ──────────────────────────────────────────────────

class TestSingleTask:
    def test_single_passing_result(self):
        results = [_make_result("bug_fixing", 0.8)]
        engine = ScoringEngine(results)
        report = engine.compute()
        assert isinstance(report, BenchmarkReport)
        assert report.final_score == pytest.approx(0.8, abs=0.01)
        assert report.total_records == 1
        assert report.total_passed == 1

    def test_single_failing_result(self):
        results = [_make_result("bug_fixing", 0.3)]
        engine = ScoringEngine(results)
        report = engine.compute()
        assert report.total_passed == 0
        assert report.total_scored == 1

    def test_single_error_result(self):
        results = [_make_error_result("bug_fixing")]
        engine = ScoringEngine(results)
        report = engine.compute()
        assert report.total_errors == 1
        assert report.final_score == 0.0


# ── Multi-task scoring ────────────────────────────────────────────────────

class TestMultiTask:
    def test_two_tasks(self):
        results = [
            _make_result("bug_fixing", 0.9),
            _make_result("code_generation", 0.7),
        ]
        engine = ScoringEngine(results)
        report = engine.compute()
        # Weighted average of 0.9 and 0.7
        assert 0.7 <= report.final_score <= 0.9

    def test_multiple_records_per_task(self):
        results = [
            _make_result("bug_fixing", 0.8),
            _make_result("bug_fixing", 0.6),
            _make_result("bug_fixing", 1.0),
        ]
        engine = ScoringEngine(results)
        report = engine.compute()
        ts = report.task_scores["bug_fixing"]
        assert ts.record_count == 3
        assert ts.scored_count == 3
        assert ts.mean_score == pytest.approx(0.8, abs=0.01)

    def test_mixed_results_and_errors(self):
        results = [
            _make_result("bug_fixing", 0.9),
            _make_error_result("bug_fixing"),
            _make_result("code_generation", 0.5),
        ]
        engine = ScoringEngine(results)
        report = engine.compute()
        assert report.total_errors == 1
        assert report.total_scored == 2


# ── Custom weights ────────────────────────────────────────────────────────

class TestCustomWeights:
    def test_custom_task_weights(self):
        results = [
            _make_result("bug_fixing", 1.0),
            _make_result("code_generation", 0.0),
        ]
        custom_weights = {"bug_fixing": 1.0, "code_generation": 0.0}
        engine = ScoringEngine(results, task_weights=custom_weights)
        report = engine.compute()
        assert report.final_score == pytest.approx(1.0, abs=0.01)

    def test_custom_pass_threshold(self):
        results = [_make_result("bug_fixing", 0.4)]
        engine = ScoringEngine(results, pass_threshold=0.3)
        report = engine.compute()
        assert report.total_passed == 1


# ── Category scores ──────────────────────────────────────────────────────

class TestCategoryScores:
    def test_categories_populated(self):
        results = [
            _make_result("bug_fixing", 0.8),
            _make_result("code_generation", 0.7),
            _make_result("code_review", 0.9),
        ]
        engine = ScoringEngine(results)
        report = engine.compute()
        assert "transformation" in report.category_scores
        assert "generation" in report.category_scores
        assert "analysis" in report.category_scores

    def test_empty_category(self):
        results = [_make_result("bug_fixing", 0.8)]
        engine = ScoringEngine(results)
        report = engine.compute()
        # generation and analysis categories have no data
        assert report.category_scores["generation"] == 0.0
        assert report.category_scores["analysis"] == 0.0


# ── Report serialisation ─────────────────────────────────────────────────

class TestReportSerialization:
    def test_to_dict(self):
        results = [_make_result("bug_fixing", 0.8)]
        engine = ScoringEngine(results)
        report = engine.compute()
        d = report.to_dict()
        assert "final_score" in d
        assert "grade" in d
        assert "task_scores" in d
        assert "category_scores" in d

    def test_task_score_to_dict(self):
        results = [_make_result("bug_fixing", 0.8)]
        engine = ScoringEngine(results)
        report = engine.compute()
        ts = report.task_scores["bug_fixing"]
        d = ts.to_dict()
        assert d["task_name"] == "bug_fixing"
        assert d["mean_score"] == pytest.approx(0.8, abs=0.01)


# ── Edge cases ────────────────────────────────────────────────────────────

class TestScoringEdgeCases:
    def test_empty_results(self):
        engine = ScoringEngine([])
        report = engine.compute()
        assert report.final_score == 0.0
        assert report.total_records == 0

    def test_all_errors(self):
        results = [_make_error_result("bug_fixing"), _make_error_result("code_generation")]
        engine = ScoringEngine(results)
        report = engine.compute()
        assert report.final_score == 0.0
        assert report.total_errors == 2

    def test_pass_rate_calculation(self):
        results = [
            _make_result("bug_fixing", 0.8),
            _make_result("bug_fixing", 0.3),
            _make_result("bug_fixing", 0.6),
        ]
        engine = ScoringEngine(results)
        report = engine.compute()
        # 2 of 3 pass (0.8 and 0.6 >= 0.5)
        ts = report.task_scores["bug_fixing"]
        assert ts.pass_rate == pytest.approx(2 / 3, abs=0.01)


# ── Constants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_task_weights_sum_to_one(self):
        assert abs(sum(TASK_WEIGHTS.values()) - 1.0) < 1e-9

    def test_grade_thresholds_descending(self):
        scores = [t[0] for t in GRADE_THRESHOLDS]
        assert scores == sorted(scores, reverse=True)

    def test_pass_threshold_reasonable(self):
        assert 0.0 < PASS_THRESHOLD < 1.0

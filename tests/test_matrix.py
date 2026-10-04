"""
End-to-end tests for the benchmark matrix configuration.
"""
import pytest
from benchmarks.matrix import (
    BENCHMARK_MATRIX,
    DIMENSION_WEIGHTS,
    get_dimensions_for_task,
    get_all_task_names,
)
from core.base import BaseDimension


# ── Matrix structure ──────────────────────────────────────────────────────

class TestMatrixStructure:
    def test_all_tasks_present(self):
        expected = {
            "bug_fixing", "code_generation", "code_review",
            "refactoring", "test_generation", "translation",
        }
        assert set(BENCHMARK_MATRIX.keys()) == expected

    def test_all_dimensions_are_classes(self):
        for task, dims in BENCHMARK_MATRIX.items():
            for dim_cls in dims:
                assert isinstance(dim_cls, type), f"{task}: {dim_cls} is not a class"

    def test_all_dimensions_inherit_base(self):
        for task, dims in BENCHMARK_MATRIX.items():
            for dim_cls in dims:
                assert issubclass(dim_cls, BaseDimension), f"{task}: {dim_cls} not a BaseDimension"

    def test_no_empty_task(self):
        for task, dims in BENCHMARK_MATRIX.items():
            assert len(dims) > 0, f"{task} has no dimensions"


# ── Dimension weights ────────────────────────────────────────────────────

class TestDimensionWeights:
    def test_weights_exist_for_all_tasks(self):
        for task in BENCHMARK_MATRIX:
            assert task in DIMENSION_WEIGHTS, f"No weights for {task}"

    def test_weights_cover_all_dimensions(self):
        for task, dims in BENCHMARK_MATRIX.items():
            dim_names = {cls.dimension_id for cls in dims}
            weight_names = set(DIMENSION_WEIGHTS[task].keys())
            assert dim_names.issubset(weight_names), (
                f"{task}: dimensions {dim_names - weight_names} have no weight"
            )

    def test_weights_are_non_negative(self):
        for task, weights in DIMENSION_WEIGHTS.items():
            for dim, w in weights.items():
                assert w >= 0.0, f"{task}/{dim}: negative weight {w}"

    def test_weights_sum_to_one(self):
        for task, weights in DIMENSION_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"{task}: weights sum to {total}"


# ── Lookup functions ──────────────────────────────────────────────────────

class TestLookupFunctions:
    def test_get_dimensions_for_known_task(self):
        dims = get_dimensions_for_task("bug_fixing")
        assert len(dims) > 0

    def test_get_dimensions_for_unknown_task(self):
        dims = get_dimensions_for_task("nonexistent_task")
        assert dims == []

    def test_get_all_task_names(self):
        names = get_all_task_names()
        assert len(names) == 6
        assert "bug_fixing" in names
        assert "translation" in names


# ── Specific task dimension mappings ──────────────────────────────────────

class TestSpecificMappings:
    def test_bug_fixing_has_functional_correctness(self):
        dims = get_dimensions_for_task("bug_fixing")
        dim_names = [cls.name for cls in dims]
        assert "Functional Correctness" in dim_names

    def test_code_review_only_has_review_dimension(self):
        dims = get_dimensions_for_task("code_review")
        assert len(dims) == 1
        assert dims[0].name == "Reference Review Similarity (RRS)"

    def test_test_generation_has_generated_test_effectiveness(self):
        dims = get_dimensions_for_task("test_generation")
        dim_names = [cls.name for cls in dims]
        assert "Generated Test Effectiveness (GTE)" in dim_names

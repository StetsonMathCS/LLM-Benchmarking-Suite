"""
benchmarks/matrix.py

Benchmark Matrix — maps each task to the dimensions it should be evaluated on.
Provides a central lookup so tasks can discover which dimensions apply to them.
"""

from benchmarks.dimensions.functional_correctness import FunctionalCorrectnessDimension
from benchmarks.dimensions.semantic_drift import SemanticDriftDimension
from benchmarks.dimensions.code_consistency import CodeConsistencyDimension
from benchmarks.dimensions.linting import LintingDimension
from benchmarks.dimensions.runtime_analysis import RuntimeAnalysisDimension
from benchmarks.dimensions.partial_transform import PartialTransformationDimension
from benchmarks.dimensions.test_pass_rate import TestPassRateDimension
from benchmarks.dimensions.vulnerabilities import VulnerabilitiesDimension
from benchmarks.dimensions.code_review import CodeReviewDimension

# Matrix Definition 
# Keys   = task names (must match BaseBenchmark.name on each task)
# Values = list of BaseDimension *classes* to instantiate at eval time

BENCHMARK_MATRIX: dict[str, list[type]] = {
    "bug_fixing" : [
        CodeConsistencyDimension,
        FunctionalCorrectnessDimension,
        LintingDimension,
        VulnerabilitiesDimension,
        SemanticDriftDimension
    ],
    "code_completion" : [
        CodeConsistencyDimension,
        FunctionalCorrectnessDimension,
        LintingDimension,
        VulnerabilitiesDimension
    ],
    "code_generation" : [
        CodeConsistencyDimension,
        FunctionalCorrectnessDimension,
        LintingDimension,
        VulnerabilitiesDimension
    ],
    "code_review" : [
        CodeReviewDimension
    ],
    "refactoring" : [
        CodeConsistencyDimension,
        FunctionalCorrectnessDimension,
        SemanticDriftDimension,
        LintingDimension,
        RuntimeAnalysisDimension,
        VulnerabilitiesDimension
    ],
    "test_generation" : [
        CodeConsistencyDimension,
        TestPassRateDimension,
        LintingDimension
    ],
    "translation":[
        CodeConsistencyDimension,
        LintingDimension,
        VulnerabilitiesDimension,
        FunctionalCorrectnessDimension
    ],
    "partial_transform" : [
        PartialTransformationDimension,
        FunctionalCorrectnessDimension
    ]
}


def get_dimensions_for_task(task_name: str) -> list[type]:
    """Return the dimension classes mapped to a given task name."""
    return BENCHMARK_MATRIX.get(task_name, [])


def get_all_task_names() -> list[str]:
    """Return every task name registered in the matrix."""
    return list(BENCHMARK_MATRIX.keys())
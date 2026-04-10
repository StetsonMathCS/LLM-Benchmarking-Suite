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
from benchmarks.dimensions.code_completion_tests import CodeCompletionTestsDimension

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
    "code_generation" : [
        CodeConsistencyDimension,
        CodeCompletionTestsDimension,
        LintingDimension,
        VulnerabilitiesDimension
    ],
    "code_review" : [
        CodeReviewDimension
    ],
    "refactoring" : [
        CodeConsistencyDimension,
        FunctionalCorrectnessDimension,
        LintingDimension,
        RuntimeAnalysisDimension,
        VulnerabilitiesDimension
    ],
    "test_generation" : [   
        CodeConsistencyDimension,
        TestPassRateDimension
    ],
    "translation":[
        CodeConsistencyDimension,
        LintingDimension,
        VulnerabilitiesDimension,
        FunctionalCorrectnessDimension
    ],
    "partial_transform" : [
        PartialTransformationDimension,
        FunctionalCorrectnessDimension,
        LintingDimension,
        VulnerabilitiesDimension,
    ]
}

# Weights per dimension
DIMENSION_WEIGHTS = {
    "bug_fixing": {
        "Functional Correctness": 0.70,
        "Linting":                0.10,
        "Code Consistency":       0.10,
        "Vulnerabilities":        0.05,
        "Semantic Drift":         0.05,
    },
    "code_generation" : {
        "Code Consistency" : 0.40,
        "Code Completion Tests" : 0.40,
        "Linting" : 0.05,
        "Vulnerabilities" : 0.05,
    },
    "code_review" : {
        "Code Review Quality" : 1.00,
    },
    "refactoring": {
        "Functional Correctness": 0.40,
        "Runtime Analysis":       0.35,
        "Code Consistency":       0.10,
        "Linting":                0.10,
        "Vulnerabilities":        0.05,
    },
    "test_generation": {
        "Code Consistency" : 0.25,
        "Test Pass Rate" : 0.75,
    },
    "translation": {
        "Code Consistency": 0.30,
        "Linting": 0.10,
        "Vulnerabilities": 0.10,
        "Functional Correctness": 0.50,
    },
    "partial_transform": {
        "Partial Transformation": 0.50,
        "Functional Correctness": 0.20,
        "Linting": 0.10,
        "Vulnerabilities": 0.10,
    }
}

def get_dimensions_for_task(task_name: str) -> list[type]:
    """Return the dimension classes mapped to a given task name."""
    return BENCHMARK_MATRIX.get(task_name, [])


def get_all_task_names() -> list[str]:
    """Return every task name registered in the matrix."""
    return list(BENCHMARK_MATRIX.keys())
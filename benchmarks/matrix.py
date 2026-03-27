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
        SemanticDriftDimension,
        RuntimeAnalysisDimension,
    ],
    "code_completion" : [
        CodeConsistencyDimension,
        CodeCompletionTestsDimension,
        LintingDimension,
        VulnerabilitiesDimension,
        RuntimeAnalysisDimension
    ],
    "code_generation" : [
        CodeConsistencyDimension,
        CodeCompletionTestsDimension,
        LintingDimension,
        VulnerabilitiesDimension,
        RuntimeAnalysisDimension
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
        FunctionalCorrectnessDimension,
        LintingDimension,
        VulnerabilitiesDimension,
        RuntimeAnalysisDimension
    ]
}

# Weights per dimension
DIMENSION_WEIGHTS = {
    "bug_fixing": {
        "Functional Correctness": 0.70,
        "Linting":                0.00,
        "Code Consistency":       0.20,
        "Vulnerabilities":        0.00,
        "Semantic Drift":         0.10,
        "Runtime Analysis":       0.00,
    },
    "code_completion" : {
        "Semantic Drift" : 0.00,
        "Code Consistency" : 0.25,
        "Code Completion Tests" : 0.75,
        "Linting" : 0.00,
        "Vulnerabilities" : 0.00,
        "Runtime Analysis" : 0.00,
    },
    "code_generation" : {
        "Code Consistency" : 0.50,
        "Code Completion Tests" : 0.50,
        "Linting" : 0.00,
        "Vulnerabilities" : 0.00,
        "Runtime Analysis" : 0.00,
    },
    "code_review" : {
        "Code Review Quality" : 1.00,
    },
    "refactoring": {
        "Functional Correctness": 0.30,
        "Semantic Drift":         0.25,
        "Runtime Analysis":       0.20,
        "Code Consistency":       0.10,
        "Linting":                0.10,
        "Vulnerabilities":        0.05,
    },
    "test_generation": {
        "Code Consistency" : 0.25,
        "Test Pass Rate" : 0.75,
        "Linting" : 0.00,
    },
    "translation": {
        "Code Consistency": 0.50,
        "Linting": 0.00,
        "Vulnerabilities": 0.00,
        "Functional Correctness": 0.50,
    },
    "partial_transform": {
        "Partial Transformation": 0.40,
        "Functional Correctness": 0.20,
        "Linting": 0.10,
        "Vulnerabilities": 0.10,
        "Runtime Analysis": 0.10,
    }
}

def get_dimensions_for_task(task_name: str) -> list[type]:
    """Return the dimension classes mapped to a given task name."""
    return BENCHMARK_MATRIX.get(task_name, [])


def get_all_task_names() -> list[str]:
    """Return every task name registered in the matrix."""
    return list(BENCHMARK_MATRIX.keys())
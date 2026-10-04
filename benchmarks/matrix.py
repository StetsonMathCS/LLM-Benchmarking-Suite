"""Frozen task-to-dimension matrix for the revised-v1 evaluator."""

from benchmarks.dimensions.functional_correctness import FunctionalCorrectnessDimension
from benchmarks.dimensions.structural_similarity import StructuralSimilarityDimension
from benchmarks.dimensions.code_consistency import CodeConsistencyDimension
from benchmarks.dimensions.linting import LintingDimension
from benchmarks.dimensions.runtime_analysis import RuntimeAnalysisDimension
from benchmarks.dimensions.generated_test_effectiveness import GeneratedTestEffectivenessDimension
from benchmarks.dimensions.vulnerabilities import VulnerabilitiesDimension
from benchmarks.dimensions.reference_review_similarity import ReferenceReviewSimilarityDimension
from benchmarks.dimensions.reference_test_success import ReferenceTestSuccessDimension
from facets.scoring_profile import DEFAULT_PROFILE, LEGACY_DIMENSION_NAMES


BENCHMARK_MATRIX: dict[str, list[type]] = {
    "bug_fixing": [
        CodeConsistencyDimension,
        FunctionalCorrectnessDimension,
        LintingDimension,
        VulnerabilitiesDimension,
        StructuralSimilarityDimension,
    ],
    "code_generation": [CodeConsistencyDimension, ReferenceTestSuccessDimension],
    "code_review": [ReferenceReviewSimilarityDimension],
    "refactoring": [
        CodeConsistencyDimension,
        FunctionalCorrectnessDimension,
        LintingDimension,
        RuntimeAnalysisDimension,
        VulnerabilitiesDimension,
    ],
    "test_generation": [CodeConsistencyDimension, GeneratedTestEffectivenessDimension],
    "translation": [
        CodeConsistencyDimension,
        LintingDimension,
        VulnerabilitiesDimension,
        FunctionalCorrectnessDimension,
    ],
}

# Compatibility symbol; keys are stable IDs, never display names.
DIMENSION_WEIGHTS = DEFAULT_PROFILE.dimension_weights


def get_dimensions_for_task(task_name: str) -> list[type]:
    return BENCHMARK_MATRIX.get(task_name, [])


def get_all_task_names() -> list[str]:
    return list(BENCHMARK_MATRIX)


def resolve_dimension_id(name_or_id: str) -> str:
    """Map explicit legacy configuration names at the import boundary."""
    return LEGACY_DIMENSION_NAMES.get(name_or_id, name_or_id)


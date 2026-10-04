import pytest

from benchmarks.dimensions.generated_test_effectiveness import GeneratedTestEffectivenessDimension
from facets.evaluation.execution import ExecutionResult, ExecutionSettings, LocalBackend


ORIGINAL = "def add(a, b):\n    return a + b\n"
REFERENCE = "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n"


@pytest.fixture
def dimension():
    settings = ExecutionSettings(backend="local", timeout_s=2)
    return GeneratedTestEffectivenessDimension(LocalBackend(settings), settings)


def evaluate(dimension, tests):
    return dimension.evaluate("python", tests, ORIGINAL, entry_point="add", reference_tests=REFERENCE)


def test_valid_strong_tests(dimension):
    result = evaluate(dimension, "from candidate import add\n\ndef test_values():\n    assert add(1, 2) == 3\n    assert add(-1, 1) == 0\n")
    assert result.status == "ok"
    assert result.details["validity"] == 1
    assert result.details["mutation_detection"] > 0
    assert result.score == result.details["validity"] * result.details["mutation_detection"]


def test_valid_weak_tests_have_lower_mutation_detection(dimension):
    result = evaluate(dimension, "from candidate import add\n\ndef test_zero():\n    assert add(0, 0) == 0\n")
    assert result.details["validity"] == 1
    assert result.details["mutation_detection"] < 1


def test_tests_wrong_on_original_reduce_validity(dimension):
    result = evaluate(dimension, "from candidate import add\n\ndef test_wrong():\n    assert add(1, 2) == 99\n")
    assert result.details["validity"] == 0
    assert result.score == 0


def test_no_tests_is_zero(dimension):
    result = evaluate(dimension, "import candidate\n")
    assert result.score == 0
    assert result.details["collected_executable_tests"] == 0


def test_collection_failure_is_not_a_mutant_kill(dimension):
    result = evaluate(dimension, "import candidate\nimport module_that_does_not_exist\n")
    assert result.score == 0
    assert result.status == "candidate_failure"


def test_complete_multiline_checker_adapter(dimension):
    tests = """\
VALUES = [(1, 2, 3), (-1, 1, 0)]
def check(candidate):
    for a, b, expected in VALUES:
        assert (
            candidate(a, b)
            == expected
        )
"""
    result = evaluate(dimension, tests)
    assert result.details["format"] == "humaneval_check"
    assert result.details["validity"] == 1


def test_parametrized_ids_are_stable(dimension):
    tests = """\
import pytest
from candidate import add
@pytest.mark.parametrize("a,b,expected", [(1,2,3), (-1,1,0)], ids=["positive", "cancel"])
def test_add(a, b, expected):
    assert add(a, b) == expected
"""
    result = evaluate(dimension, tests)
    assert any("positive" in test_id for test_id in result.details["baseline_passing_test_ids"])


def test_candidate_shadowing_rejected(dimension):
    tests = "def add(a, b): return a + b\n\ndef test_copy(): assert add(1, 2) == 3\n"
    result = evaluate(dimension, tests)
    assert result.status == "candidate_failure"
    assert "shadowing" in result.details["diagnostic"]


def test_unequal_collected_ids_do_not_kill_mutant(dimension):
    tests = """\
import candidate
if candidate.add(1, 2) == 3:
    def test_baseline_only(): assert True
else:
    def test_mutant_only(): assert False
"""
    result = evaluate(dimension, tests)
    assert any(item["test_ids_match"] is False and item["detected"] is False for item in result.details["mutants"])


class BrokenBackend:
    def run_python_files(self, files, entry_file, timeout_s=None):
        return ExecutionResult(backend="broken", diagnostic="executor unavailable")


def test_infrastructure_error_is_distinct():
    settings = ExecutionSettings(backend="local")
    dimension = GeneratedTestEffectivenessDimension(BrokenBackend(), settings)
    result = evaluate(dimension, "from candidate import add\ndef test_add(): assert add(1,2)==3")
    assert result.status == "infrastructure_error"


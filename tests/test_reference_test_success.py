import pytest

from benchmarks.dimensions.reference_test_success import ReferenceTestSuccessDimension
from facets.evaluation.execution import ExecutionSettings, LocalBackend


@pytest.fixture
def dimension():
    settings = ExecutionSettings(backend="local", timeout_s=0.5)
    return ReferenceTestSuccessDimension(LocalBackend(settings), settings)


CHECK = """\
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(-1, 1) == 0
"""


@pytest.mark.parametrize("candidate", [
    "def add(a, b):\n    return a - b\n",
    "def other(a, b):\n    return a + b\n",
    "",
    "def add(a, b\n    return 3\n",
    "def add(a, b):\n    raise RuntimeError('boom')\n",
])
def test_candidate_failures_score_zero(dimension, candidate):
    result = dimension.evaluate("python", candidate, test=CHECK, entry_point="add")
    assert result.score == 0
    assert result.status == "candidate_failure"


def test_correct_candidate_invokes_complete_checker(dimension):
    result = dimension.evaluate("python", "def add(a, b):\n    return a + b\n", test=CHECK, entry_point="add")
    assert result.score == 1
    assert result.details["checks_invoked"] is True
    assert result.details["test_count"] is None
    assert len(result.details["test_hash"]) == 64


def test_timeout_is_candidate_failure(dimension):
    candidate = "def add(a, b):\n    while True:\n        pass\n"
    result = dimension.evaluate("python", candidate, test=CHECK, entry_point="add", timeout_s=0.1)
    assert result.score == 0
    assert result.details["timeout"] is True


def test_multiline_setup_and_loop_are_preserved(dimension):
    checker = """\
import math
VALUES = [(1, 2, 3), (2, 5, 7)]
def check(candidate):
    for left, right, expected in VALUES:
        assert (
            candidate(left, right)
            == expected
        )
"""
    result = dimension.evaluate("python", "def add(a, b):\n    return a + b\n", test=checker, entry_point="add")
    assert result.score == 1


def test_defining_checker_without_invocation_never_passes(dimension):
    checker = "def check(candidate):\n    assert candidate() == 42\n"
    result = dimension.evaluate("python", "def answer():\n    return 0\n", test=checker, entry_point="answer")
    assert result.score == 0
    assert result.details["checks_invoked"] is True


def test_malformed_reference_is_infrastructure_error(dimension):
    result = dimension.evaluate("python", "def add(a, b): return a+b", test="x = 1", entry_point="add")
    assert result.status == "infrastructure_error"


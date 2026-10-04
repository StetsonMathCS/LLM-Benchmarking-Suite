"""Reference Test Success (RTS) for trusted HumanEval-style checkers."""

from __future__ import annotations

import ast
import hashlib
import keyword
import secrets
from typing import Optional

from core.base import BaseDimension, DimensionResult
from facets.evaluation.execution import ExecutionSettings, default_settings, get_backend


INVOKED_MARKER = "FACETS_REFERENCE_CHECK_INVOKED"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReferenceTestSuccessDimension(BaseDimension):
    """Binary success on the complete trusted reference checker.

    Candidate code and trusted tests are imported as separate modules so a
    candidate definition named ``check`` cannot replace the trusted binding.
    """

    dimension_id = "reference_test_success"
    name = "Reference Test Success (RTS)"
    description = "Observed success on the complete trusted reference checker."

    def __init__(self, backend=None, settings: Optional[ExecutionSettings] = None):
        self.settings = settings or default_settings()
        self.backend = backend or get_backend(self.settings)

    def _result(self, score: float, *, status: str, details: dict) -> DimensionResult:
        return DimensionResult(
            dimension_id=self.dimension_id,
            dimension_name=self.name,
            score=score,
            passed=score == 1.0,
            status=status,
            details=details,
        )

    def evaluate(self, language: str, generated_code: str, **kwargs) -> DimensionResult:
        reference_test = kwargs.get("test")
        entry_point = kwargs.get("entry_point")
        dataset_hash = kwargs.get("dataset_hash")
        base = {
            "format": "humaneval_check",
            "checks_invoked": False,
            "entry_point": entry_point,
            "test_hash": _sha256(reference_test) if isinstance(reference_test, str) else None,
            "dataset_hash": dataset_hash,
            "test_count": None,
        }
        if language != "python":
            return self._result(0.0, status="infrastructure_error", details={
                **base, "diagnostic": "RTS has no adapter for this mandatory format/language",
            })
        if not isinstance(reference_test, str) or not reference_test.strip():
            return self._result(0.0, status="infrastructure_error", details={
                **base, "diagnostic": "missing trusted reference test body",
            })
        if not isinstance(entry_point, str) or not entry_point.isidentifier() or keyword.iskeyword(entry_point):
            return self._result(0.0, status="infrastructure_error", details={
                **base, "diagnostic": "entry_point must be a Python identifier",
            })
        if not isinstance(generated_code, str) or not generated_code.strip():
            return self._result(0.0, status="candidate_failure", details={
                **base, "diagnostic": "empty candidate",
            })
        try:
            ast.parse(generated_code, filename="candidate.py")
        except SyntaxError as exc:
            return self._result(0.0, status="candidate_failure", details={
                **base, "diagnostic": f"candidate syntax error at line {exc.lineno}: {exc.msg}",
                "compilation_status": "failed",
            })
        try:
            tree = ast.parse(reference_test, filename="reference_tests.py")
        except SyntaxError as exc:
            return self._result(0.0, status="infrastructure_error", details={
                **base, "diagnostic": f"trusted reference syntax error at line {exc.lineno}: {exc.msg}",
            })
        check_defs = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check"
        ]
        if len(check_defs) != 1 or len(check_defs[0].args.args) < 1:
            return self._result(0.0, status="infrastructure_error", details={
                **base, "diagnostic": "expected exactly one def check(candidate) reference checker",
            })

        invocation_marker = f"{INVOKED_MARKER}:{secrets.token_hex(16)}"
        # Import the trusted module first and retain its module binding.  The
        # candidate is deliberately a distinct module: a candidate's own
        # ``check`` symbol must never become the trusted checker by accident.
        harness = f"""\
import reference_tests as _facets_reference_module
import candidate as _facets_candidate_module

_candidate = getattr(_facets_candidate_module, {entry_point!r}, None)
if not callable(_candidate):
    raise TypeError({('entry point ' + entry_point + ' is missing or not callable')!r})
print({invocation_marker!r}, flush=True)
_facets_reference_module.check(_candidate)
"""
        execution = self.backend.run_python_files(
            {
                "candidate.py": generated_code,
                "reference_tests.py": reference_test,
                "run_reference.py": harness,
            },
            "run_reference.py",
            timeout_s=float(kwargs.get("timeout_s", self.settings.timeout_s)),
        )
        invoked = invocation_marker in execution.stdout
        details = {
            **base,
            "checks_invoked": invoked,
            "execution": execution.to_dict(),
            "exit_status": execution.exit_code,
            "timeout": execution.timed_out,
            "compilation_status": "passed",
        }
        if execution.succeeded and invoked:
            details["diagnostic"] = "trusted reference checker completed"
            return self._result(1.0, status="ok", details=details)
        if execution.exit_code is None and not execution.timed_out and execution.diagnostic:
            details["diagnostic"] = execution.diagnostic
            return self._result(0.0, status="infrastructure_error", details=details)
        if execution.timed_out:
            details["diagnostic"] = "candidate timed out during trusted reference check"
        elif not invoked:
            details["diagnostic"] = "candidate failed before trusted checks could be invoked"
        else:
            details["diagnostic"] = "trusted reference checker rejected candidate"
        return self._result(0.0, status="candidate_failure", details=details)

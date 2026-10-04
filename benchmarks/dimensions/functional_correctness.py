"""Observed functional behavior on explicitly supplied workloads/outputs."""

from __future__ import annotations

import re
from typing import Optional

from core.base import BaseDimension, DimensionResult
from facets.evaluation.execution import default_settings, get_backend


def _normalize_output(text: str, language: str) -> str:
    """Conservative cross-language normalization for scalar console output."""
    value = text.replace("\r\n", "\n").rstrip("\n")
    if language == "javascript":
        value = re.sub(r"\btrue\b", "True", value)
        value = re.sub(r"\bfalse\b", "False", value)
        value = re.sub(r"\bnull\b", "None", value)
        value = re.sub(r"\bundefined\b", "None", value)
    return value


class FunctionalCorrectnessDimension(BaseDimension):
    dimension_id = "functional_correctness"
    name = "Functional Correctness"
    description = "Observed behavior on supplied tests or inputs; not a proof of correctness."

    def __init__(self, backend=None, settings=None):
        self.settings = settings or default_settings()
        self.backend = backend or get_backend(self.settings)

    def _run(self, language: str, code: str):
        if language == "python":
            return self.backend.run_python_files({"candidate.py": code}, "candidate.py")
        if language == "javascript":
            return self.backend.run_files(
                {"candidate.js": code}, ["node", "--max-old-space-size=128", "candidate.js"]
            )
        if language == "cpp":
            return self.backend.run_files(
                {"candidate.cpp": code}, ["sh", "-c", "g++ candidate.cpp -O2 -o candidate && ./candidate"]
            )
        return None

    def evaluate(
        self, language: str, generated_code: str, original_code: Optional[str] = None, **kwargs
    ) -> DimensionResult:
        expected_output_present = "expected_output" in kwargs and kwargs.get("expected_output") is not None
        expected_console_present = (
            "expected_console_output" in kwargs and kwargs.get("expected_console_output") is not None
        )
        generated_language = kwargs.get("generated_code_language", language)
        workload = kwargs.get("test_harness", "")

        if expected_console_present:
            expected = str(kwargs.get("expected_console_output"))
            candidate_program = generated_code + ("\n" + workload if workload else "")
            generated = self._run(generated_language, candidate_program)
            reference = None
        elif expected_output_present and not kwargs.get("expected_output_is_implementation", False):
            expected = str(kwargs.get("expected_output"))
            generated = self._run(generated_language, generated_code)
            reference = None
        elif original_code is not None:
            reference = self._run(language, original_code + ("\n" + workload if workload else ""))
            generated = self._run(generated_language, generated_code + ("\n" + workload if workload else ""))
            expected = reference.stdout if reference and reference.succeeded else ""
        else:
            return DimensionResult(
                self.name, 0.0, False,
                {"diagnostic": "no expected behavior, trusted reference, or workload supplied"},
                dimension_id=self.dimension_id, status="infrastructure_error",
            )

        if generated is None:
            return DimensionResult(
                self.name, 0.0, False,
                {"diagnostic": f"unsupported generated language: {generated_language}"},
                dimension_id=self.dimension_id, status="infrastructure_error",
            )
        if reference is not None and not reference.succeeded:
            return DimensionResult(
                self.name, 0.0, False,
                {"diagnostic": "trusted reference execution failed", "reference_execution": reference.to_dict()},
                dimension_id=self.dimension_id, status="infrastructure_error",
            )
        if not generated.succeeded:
            infrastructure = generated.exit_code is None and not generated.timed_out
            return DimensionResult(
                self.name, 0.0, False,
                {"diagnostic": generated.diagnostic or "candidate execution failed", "generated_execution": generated.to_dict()},
                dimension_id=self.dimension_id,
                status="infrastructure_error" if infrastructure else "candidate_failure",
            )

        normalized_expected = _normalize_output(expected, language)
        normalized_generated = _normalize_output(generated.stdout, generated_language)
        match = normalized_expected == normalized_generated
        return DimensionResult(
            self.name,
            1.0 if match else 0.0,
            match,
            {
                "expected_output": expected,
                "generated_output": generated.stdout,
                "match": match,
                "normalization": "line endings, trailing final newlines, and JS scalar literals only",
                "generated_execution": generated.to_dict(),
                "reference_execution": reference.to_dict() if reference else None,
            },
            dimension_id=self.dimension_id,
            status="ok" if match else "candidate_failure",
        )

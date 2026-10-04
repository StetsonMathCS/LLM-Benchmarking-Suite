"""Measured workload time/memory with correctness gating."""

from __future__ import annotations

import json
import math
import platform
from statistics import median
from typing import Optional

from core.base import BaseDimension, DimensionResult
from facets.evaluation.execution import ExecutionSettings, default_settings, get_backend


RESULT_MARKER = "FACETS_RUNTIME_RESULT="


def relative_efficiency_component(baseline: Optional[float], generated: Optional[float]) -> Optional[float]:
    """Return min(1, baseline/generated); unavailable/nonpositive values stay unavailable."""
    if baseline is None or generated is None or baseline <= 0 or generated <= 0:
        return None
    return min(1.0, baseline / generated)


def combined_efficiency_score(time_component: Optional[float], memory_component: Optional[float]) -> Optional[float]:
    if time_component is None or memory_component is None:
        return None
    return math.sqrt(time_component * memory_component)


def _measurement_program(source: str, workload: str, warmup: int, repeats: int) -> str:
    return f'''\
import contextlib
import io
import json
import resource
import statistics
import time

exec(compile({source!r}, "evaluated_source.py", "exec"), globals())
_workload = compile({workload!r}, "facets_workload.py", "exec")
_sink = io.StringIO()
for _ in range({warmup}):
    with contextlib.redirect_stdout(_sink):
        exec(_workload, globals())
_samples = []
for _ in range({repeats}):
    _start = time.perf_counter_ns()
    with contextlib.redirect_stdout(_sink):
        exec(_workload, globals())
    _samples.append(time.perf_counter_ns() - _start)
_peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
_payload = {{
    "time_samples_ns": _samples,
    "median_time_ns": statistics.median(_samples),
    "peak_memory_kib": _peak_kib,
    "workload_invocations": {warmup + repeats},
}}
print({RESULT_MARKER!r} + json.dumps(_payload, separators=(",", ":")))
'''


class RuntimeAnalysisDimension(BaseDimension):
    dimension_id = "runtime_analysis"
    name = "Runtime Analysis"
    description = "Measured deterministic workload time and peak memory; not asymptotic complexity."

    def __init__(self, backend=None, settings: Optional[ExecutionSettings] = None):
        self.settings = settings or default_settings()
        self.backend = backend or get_backend(self.settings)

    @staticmethod
    def ratio_score(baseline: float, generated: float) -> float:
        value = relative_efficiency_component(baseline, generated)
        return 0.0 if value is None else value

    def _execute_once(self, source: str, workload: str, timeout_s: float):
        return self.backend.run_python_files(
            {"evaluated_source.py": source, "run_once.py": source + "\n" + workload},
            "run_once.py",
            timeout_s=timeout_s,
        )

    def _measure(self, source: str, workload: str, warmup: int, repeats: int, timeout_s: float) -> tuple[dict, dict]:
        execution = self.backend.run_python_files(
            {"measure.py": _measurement_program(source, workload, warmup, repeats)},
            "measure.py",
            timeout_s=timeout_s,
        )
        payload: dict = {}
        for line in execution.stdout.splitlines():
            if line.startswith(RESULT_MARKER):
                try:
                    payload = json.loads(line[len(RESULT_MARKER):])
                except json.JSONDecodeError:
                    payload = {}
        return payload, execution.to_dict()

    def evaluate(
        self, language: str, generated_code: str, original_code: Optional[str] = None, **kwargs
    ) -> DimensionResult:
        workload = kwargs.get("test_harness")
        expected = kwargs.get("expected_console_output")
        baseline_code = kwargs.get("expected_output") if kwargs.get("expected_output") is not None else original_code
        warmup = int(kwargs.get("runtime_warmup", 1))
        repeats = int(kwargs.get("runtime_repeats", 7))
        timeout_s = float(kwargs.get("timeout_s", self.settings.timeout_s))
        base_details = {
            "method": "in-child perf_counter_ns median; per-process ru_maxrss peak",
            "units": {"time": "ns", "memory": "KiB"},
            "warmup": warmup,
            "repeats": repeats,
            "baseline": "trusted expected refactored implementation" if kwargs.get("expected_output") is not None else "original implementation",
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "backend": self.settings.backend},
        }
        if language != "python":
            return DimensionResult(self.name, 0.0, False, {
                **base_details, "diagnostic": "revised-v1 runtime measurement currently supports Python only",
            }, dimension_id=self.dimension_id, status="not_applicable", applicable=False)
        if not isinstance(workload, str) or not workload.strip():
            return DimensionResult(self.name, 0.0, False, {
                **base_details, "diagnostic": "missing deterministic workload/test harness",
            }, dimension_id=self.dimension_id, status="infrastructure_error")
        if not isinstance(baseline_code, str) or not baseline_code.strip():
            return DimensionResult(self.name, 0.0, False, {
                **base_details, "diagnostic": "missing baseline implementation",
            }, dimension_id=self.dimension_id, status="infrastructure_error")
        if expected is None:
            return DimensionResult(self.name, 0.0, False, {
                **base_details, "diagnostic": "missing expected workload output for correctness gate",
            }, dimension_id=self.dimension_id, status="infrastructure_error")

        baseline_once = self._execute_once(baseline_code, workload, timeout_s)
        generated_once = self._execute_once(generated_code, workload, timeout_s)
        expected_text = str(expected).rstrip("\r\n")
        baseline_correct = baseline_once.succeeded and baseline_once.stdout.rstrip("\r\n") == expected_text
        generated_correct = generated_once.succeeded and generated_once.stdout.rstrip("\r\n") == expected_text
        base_details["correctness_gate"] = {
            "expected": expected,
            "baseline": baseline_once.to_dict(),
            "generated": generated_once.to_dict(),
            "baseline_passed": baseline_correct,
            "generated_passed": generated_correct,
        }
        if not baseline_correct:
            return DimensionResult(self.name, 0.0, False, {
                **base_details, "diagnostic": "trusted baseline/workload contract failed",
            }, dimension_id=self.dimension_id, status="infrastructure_error")
        if not generated_correct:
            return DimensionResult(self.name, 0.0, False, {
                **base_details, "diagnostic": "candidate failed associated correctness workload",
            }, dimension_id=self.dimension_id, status="candidate_failure")

        baseline, baseline_exec = self._measure(baseline_code, workload, warmup, repeats, timeout_s)
        generated, generated_exec = self._measure(generated_code, workload, warmup, repeats, timeout_s)
        baseline_time = baseline.get("median_time_ns")
        generated_time = generated.get("median_time_ns")
        baseline_memory = baseline.get("peak_memory_kib")
        generated_memory = generated.get("peak_memory_kib")
        time_component = relative_efficiency_component(baseline_time, generated_time)
        memory_component = relative_efficiency_component(baseline_memory, generated_memory)
        score = combined_efficiency_score(time_component, memory_component)
        details = {
            **base_details,
            "baseline_measurement": baseline,
            "generated_measurement": generated,
            "baseline_execution": baseline_exec,
            "generated_execution": generated_exec,
            "raw_time_ratio": (baseline_time / generated_time) if baseline_time and generated_time else None,
            "raw_memory_ratio": (baseline_memory / generated_memory) if baseline_memory and generated_memory else None,
            "time_component": time_component,
            "memory_component": memory_component,
        }
        if score is None:
            details["diagnostic"] = "time or memory measurement unavailable/nonpositive"
            return DimensionResult(self.name, 0.0, False, details, dimension_id=self.dimension_id, status="infrastructure_error")
        details["diagnostic"] = "correct workload executed and measured"
        return DimensionResult(self.name, score, True, details, dimension_id=self.dimension_id, status="ok")

"""Generated Test Effectiveness (GTE): validity times mutation detection."""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Optional

from core.base import BaseDimension, DimensionResult
from facets.evaluation.execution import ExecutionSettings, default_settings, get_backend
from benchmarks.dimensions.reference_test_success import ReferenceTestSuccessDimension


PYTEST_PLUGIN = r'''\
import json

_records = {}
_collection_errors = []
FACETS_PAYLOAD = None

# The payload rides on stdout, and stdout is truncated to OUTPUT_LIMIT keeping
# its head. Anything emitted after the marker is therefore unreachable, so the
# payload must stay small enough to fit ahead of pytest's own output.
_DIAGNOSTIC_CHARS = 300
_MAX_DIAGNOSTICS = 20

def _diagnostic(report):
    if sum(1 for item in _records.values() if "diagnostic" in item) >= _MAX_DIAGNOSTICS:
        return "<diagnostic omitted: per-record diagnostic budget exhausted>"
    return str(report.longrepr)[:_DIAGNOSTIC_CHARS]

def pytest_collectreport(report):
    if report.failed:
        if len(_collection_errors) < _MAX_DIAGNOSTICS:
            _collection_errors.append({"nodeid": report.nodeid, "diagnostic": str(report.longrepr)[:_DIAGNOSTIC_CHARS]})

def pytest_runtest_logreport(report):
    if report.when not in ("setup", "call", "teardown"):
        return
    current = _records.setdefault(report.nodeid, {"id": report.nodeid, "phases": {}})
    outcome = report.outcome
    wasxfail = getattr(report, "wasxfail", None)
    if wasxfail:
        outcome = "xfail" if report.skipped else "xpass"
    current["phases"][report.when] = outcome
    current["duration_s"] = current.get("duration_s", 0.0) + float(getattr(report, "duration", 0.0))
    if report.failed:
        current["diagnostic"] = _diagnostic(report)

def pytest_sessionfinish(session, exitstatus):
    global FACETS_PAYLOAD
    outcomes = []
    for nodeid in sorted(_records):
        item = _records[nodeid]
        phases = item["phases"]
        values = set(phases.values())
        if "failed" in values:
            status = "error" if phases.get("setup") == "failed" or phases.get("teardown") == "failed" else "failed"
        elif "xpass" in values:
            status = "xpass"
        elif "xfail" in values:
            status = "xfail"
        elif "skipped" in values:
            status = "skipped"
        elif phases.get("call") == "passed":
            status = "passed"
        else:
            status = "error"
        outcomes.append({**item, "status": status})
    payload = {"exit_code": int(exitstatus), "tests": outcomes, "collection_errors": _collection_errors}
    FACETS_PAYLOAD = payload
'''

PYTEST_RUNNER = r'''\
import json
import pytest
import facets_pytest_plugin
# --tb=no: pytest otherwise echoes every failure message on stdout, which the host
# truncates before reaching the trailing result marker.
status = pytest.main(["-q", "--tb=no", "-p", "no:cacheprovider", "-p", "facets_pytest_plugin", "test_generated.py"])
if facets_pytest_plugin.FACETS_PAYLOAD is not None:
    print("FACETS_TEST_RESULT=" + json.dumps(facets_pytest_plugin.FACETS_PAYLOAD, separators=(",", ":")), flush=True)
raise SystemExit(status)
'''


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _top_level_bound_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


class _Mutator(ast.NodeTransformer):
    def __init__(self, entry_point: str, operator: str):
        self.entry_point = entry_point
        self.operator = operator
        self.in_target = False
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        previous = self.in_target
        self.in_target = node.name == self.entry_point
        result = self.generic_visit(node)
        self.in_target = previous
        return result

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Return(self, node: ast.Return):
        node = self.generic_visit(node)
        if self.in_target and not self.changed:
            replacement = {
                "return_none": ast.Constant(None),
                "return_zero": ast.Constant(0),
                "return_false": ast.Constant(False),
            }.get(self.operator)
            if replacement is not None and ast.dump(node.value) != ast.dump(replacement):
                node.value = replacement
                self.changed = True
        return node

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        if self.in_target and not self.changed and self.operator == "flip_comparison" and node.ops:
            replacements = {
                ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
                ast.Lt: ast.GtE, ast.LtE: ast.Gt,
                ast.Gt: ast.LtE, ast.GtE: ast.Lt,
                ast.Is: ast.IsNot, ast.IsNot: ast.Is,
                ast.In: ast.NotIn, ast.NotIn: ast.In,
            }
            replacement = replacements.get(type(node.ops[0]))
            if replacement:
                node.ops[0] = replacement()
                self.changed = True
        return node

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        if self.in_target and not self.changed and self.operator == "negate_condition":
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.changed = True
        return node


def build_mutants(source: str, entry_point: str, maximum: int = 5) -> list[dict]:
    """Build a deterministic, bounded set of first-site AST mutants."""
    mutants: list[dict] = []
    seen: set[str] = set()
    for operator in ("return_none", "return_zero", "return_false", "flip_comparison", "negate_condition"):
        tree = ast.parse(source)
        mutator = _Mutator(entry_point, operator)
        changed = mutator.visit(tree)
        ast.fix_missing_locations(changed)
        if not mutator.changed:
            continue
        mutated = ast.unparse(changed) + "\n"
        digest = _hash(mutated)
        if digest in seen:
            continue
        seen.add(digest)
        mutants.append({"operator": operator, "source": mutated, "hash": digest})
        if len(mutants) >= maximum:
            break
    return mutants


class GeneratedTestEffectivenessDimension(BaseDimension):
    dimension_id = "generated_test_effectiveness"
    name = "Generated Test Effectiveness (GTE)"
    description = "Validity on correct code multiplied by detection of trusted-test-validated mutants."

    def __init__(self, backend=None, settings: Optional[ExecutionSettings] = None):
        self.settings = settings or default_settings()
        self.backend = backend or get_backend(self.settings)

    def _result(self, score: float, status: str, details: dict) -> DimensionResult:
        return DimensionResult(
            dimension_id=self.dimension_id,
            dimension_name=self.name,
            score=score,
            passed=status == "ok" and score > 0.0,
            status=status,
            details=details,
        )

    def _adapt_tests(self, generated: str, entry_point: str) -> tuple[Optional[str], str, Optional[str]]:
        try:
            tree = ast.parse(generated, filename="generated_tests.py")
        except SyntaxError as exc:
            return None, "unknown", f"generated test syntax error at line {exc.lineno}: {exc.msg}"
        if entry_point in _top_level_bound_names(tree):
            return None, "unknown", "candidate shadowing: generated suite defines the candidate entry point"
        check_defs = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check"
        ]
        if check_defs:
            if len(check_defs) != 1 or len(check_defs[0].args.args) < 1:
                return None, "humaneval_check", "malformed check(candidate) suite"
            adapted = generated.rstrip() + (
                "\n\ndef test_facets_humaneval_check():\n"
                "    import candidate as _facets_candidate\n"
                f"    check(_facets_candidate.{entry_point})\n"
            )
            return adapted, "humaneval_check", None

        imports_candidate = False
        for node in tree.body:
            if isinstance(node, ast.Import) and any(alias.name == "candidate" for alias in node.names):
                imports_candidate = True
            if isinstance(node, ast.ImportFrom) and node.module == "candidate":
                imports_candidate = True
        if not imports_candidate:
            return None, "pytest_or_unittest", (
                "generated suite must import the supplied candidate module; bare/copied implementations are rejected"
            )
        has_unittest = any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and ((isinstance(node, ast.ImportFrom) and node.module == "unittest")
                 or (isinstance(node, ast.Import) and any(alias.name == "unittest" for alias in node.names)))
            for node in tree.body
        )
        return generated, "unittest" if has_unittest else "pytest", None

    def _run_suite(self, candidate: str, tests: str, timeout_s: float) -> dict:
        execution = self.backend.run_python_files(
            {
                "candidate.py": candidate,
                "test_generated.py": tests,
                "facets_pytest_plugin.py": PYTEST_PLUGIN,
                "run_tests.py": PYTEST_RUNNER,
            },
            "run_tests.py",
            timeout_s=timeout_s,
        )
        payload = None
        # The plugin emits one machine-readable marker after collection/run.
        marker = "FACETS_TEST_RESULT="
        for line in execution.stdout.splitlines():
            if line.startswith(marker):
                try:
                    payload = json.loads(line[len(marker):])
                except json.JSONDecodeError:
                    payload = None
        structured_missing = payload is None
        if payload is None:
            payload = {"tests": [], "collection_errors": [], "exit_code": execution.exit_code}
        return {"execution": execution.to_dict(), "structured_results_missing": structured_missing, **payload}

    def evaluate(
        self, language: str, generated_tests: str, original_code: str, **kwargs
    ) -> DimensionResult:
        entry_point = kwargs.get("entry_point", "")
        reference_tests = kwargs.get("reference_tests", "")
        timeout_s = float(kwargs.get("timeout_s", self.settings.timeout_s))
        common = {
            "entry_point": entry_point,
            "generated_test_hash": _hash(generated_tests or ""),
            "original_hash": _hash(original_code or ""),
            "reference_test_hash": _hash(reference_tests or ""),
            "validity_policy": "skip and expected-failure outcomes are excluded from executable-test denominator",
            "mutation_policy": "only baseline-passing test IDs can detect a trusted-test-distinguished mutant",
        }
        if language != "python":
            return self._result(0.0, "infrastructure_error", {**common, "diagnostic": "GTE supports Python only"})
        if not all(isinstance(value, str) and value.strip() for value in (generated_tests, original_code, entry_point, reference_tests)):
            return self._result(0.0, "infrastructure_error", {**common, "diagnostic": "missing generated suite, original code, entry point, or trusted reference tests"})
        try:
            ast.parse(original_code)
        except SyntaxError as exc:
            return self._result(0.0, "infrastructure_error", {**common, "diagnostic": f"known-correct source is invalid: {exc}"})

        adapted, format_name, error = self._adapt_tests(generated_tests, entry_point)
        common["format"] = format_name
        if error:
            return self._result(0.0, "candidate_failure", {**common, "diagnostic": error})

        reference_dimension = ReferenceTestSuccessDimension(backend=self.backend, settings=self.settings)
        baseline_reference = reference_dimension.evaluate(
            "python", original_code, test=reference_tests, entry_point=entry_point, timeout_s=timeout_s
        )
        if baseline_reference.status != "ok":
            return self._result(0.0, "infrastructure_error", {
                **common,
                "diagnostic": "known-correct baseline failed trusted reference validation",
                "baseline_reference": baseline_reference.to_dict(),
            })

        baseline = self._run_suite(original_code, adapted or "", timeout_s)
        # A suite the wall clock kills emits no marker, so missing structured
        # results must be read alongside the exit status. A timeout is the
        # candidate's own doing, not a harness fault, and belongs to the
        # candidate_failure path below.
        if baseline["execution"].get("timed_out"):
            return self._result(0.0, "candidate_failure", {
                **common, "diagnostic": "generated suite timed out", "baseline": baseline,
            })
        if baseline.get("structured_results_missing"):
            return self._result(0.0, "infrastructure_error", {
                **common, "diagnostic": "structured pytest reporting unavailable", "baseline": baseline,
            })
        if baseline.get("collection_errors"):
            return self._result(0.0, "candidate_failure", {
                **common, "diagnostic": "generated suite failed collection", "baseline": baseline,
            })
        outcomes = {item["id"]: item["status"] for item in baseline.get("tests", [])}
        executable = {key: value for key, value in outcomes.items() if value not in {"skipped", "xfail"}}
        passing_ids = sorted(key for key, value in executable.items() if value in {"passed", "xpass"})
        validity = len(passing_ids) / len(executable) if executable else 0.0

        built = build_mutants(original_code, entry_point)
        eligible: list[dict] = []
        excluded: list[dict] = []
        for mutant in built:
            validation = reference_dimension.evaluate(
                "python", mutant["source"], test=reference_tests, entry_point=entry_point, timeout_s=timeout_s
            )
            summary = {"operator": mutant["operator"], "hash": mutant["hash"], "reference_status": validation.status}
            if validation.status == "candidate_failure":
                eligible.append(mutant)
            else:
                reason = "not distinguished by trusted tests" if validation.status == "ok" else "invalid/infrastructure failure"
                excluded.append({**summary, "reason": reason})
        pool_hash = _hash("\n".join(mutant["hash"] for mutant in eligible))
        if not eligible:
            return self._result(0.0, "infrastructure_error", {
                **common,
                "diagnostic": "no eligible trusted-test-distinguished mutants for mandatory GTE record",
                "validity": validity,
                "baseline": baseline,
                "excluded_mutants": excluded,
                "mutant_pool_hash": pool_hash,
            })

        killed = 0
        diagnostics: list[dict] = []
        for mutant in eligible:
            result = self._run_suite(mutant["source"], adapted or "", timeout_s)
            mutant_outcomes = {item["id"]: item["status"] for item in result.get("tests", [])}
            same_ids = set(mutant_outcomes) == set(outcomes)
            infrastructure = bool(result.get("structured_results_missing") or result["execution"].get("timed_out") or result.get("collection_errors") or not same_ids)
            detected_by = [] if infrastructure else sorted(
                test_id for test_id in passing_ids
                if mutant_outcomes.get(test_id) in {"failed", "error"}
            )
            if detected_by:
                killed += 1
            diagnostics.append({
                "operator": mutant["operator"],
                "hash": mutant["hash"],
                "detected": bool(detected_by),
                "detected_by": detected_by,
                "infrastructure_error": infrastructure,
                "test_ids_match": same_ids,
                "outcomes": mutant_outcomes,
                "execution": result["execution"],
            })
        mutation_detection = killed / len(eligible)
        score = validity * mutation_detection
        return self._result(score, "ok", {
            **common,
            "validity": validity,
            "mutation_detection": mutation_detection,
            "gte": score,
            "collected_executable_tests": len(executable),
            "baseline_passing_test_ids": passing_ids,
            "baseline_outcomes": outcomes,
            "eligible_mutants": len(eligible),
            "detected_mutants": killed,
            "mutant_pool_hash": pool_hash,
            "mutants": diagnostics,
            "excluded_mutants": excluded,
        })

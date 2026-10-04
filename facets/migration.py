"""Restricted adapters for legacy repr-string reports (never uses eval)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from facets.config import fingerprint
from facets.runstore import atomic_json
from facets.scoring_profile import LEGACY_DIMENSION_NAMES


class LegacyParseError(ValueError):
    pass


def _restricted(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_restricted(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_restricted(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_restricted(key): _restricted(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _restricted(node.operand)
        if isinstance(value, (int, float)):
            return -value
    if isinstance(node, ast.Call):
        function = node.func.id if isinstance(node.func, ast.Name) else None
        if function not in {"DimensionResult", "LLMResponse"}:
            raise LegacyParseError(f"call is not permitted: {function or 'attribute'}")
        values = {}
        for keyword_node in node.keywords:
            if not keyword_node.arg:
                continue
            if keyword_node.arg == "raw_response":
                values[keyword_node.arg] = {"unrecoverable_repr": ast.unparse(keyword_node.value)}
            else:
                values[keyword_node.arg] = _restricted(keyword_node.value)
        positional = [_restricted(arg) for arg in node.args]
        if function == "DimensionResult":
            fields = ("dimension_name", "score", "passed", "details", "issues")
        else:
            fields = ("content", "model", "provider", "prompt_tokens", "completion_tokens", "latency_ms", "raw_response", "error")
        for name, value in zip(fields, positional):
            values.setdefault(name, value)
        values["legacy_repr_type"] = function
        return values
    raise LegacyParseError(f"unsupported legacy repr syntax: {type(node).__name__}")


def parse_legacy_repr(text: str) -> dict:
    try:
        expression = ast.parse(text, mode="eval").body
        value = _restricted(expression)
    except (SyntaxError, LegacyParseError) as exc:
        raise LegacyParseError(str(exc)) from exc
    if not isinstance(value, dict):
        raise LegacyParseError("legacy representation did not decode to an object")
    return value


def migrate_legacy_report(source: Path, output: Path) -> Path:
    """Recover verifiable fields without presenting old CGT/TPR as revised scores."""
    report = json.loads(source.read_text(encoding="utf-8"))
    migrated = []
    diagnostics = []
    for index, record in enumerate(report.get("results", [])):
        new_record = {key: value for key, value in record.items() if key not in {"details", "llm_response"}}
        dimensions = {}
        for legacy_name, raw in (record.get("details") or {}).items():
            try:
                parsed = parse_legacy_repr(raw) if isinstance(raw, str) else raw
                stable_id = LEGACY_DIMENSION_NAMES.get(legacy_name, legacy_name)
                dimensions[stable_id] = {
                    "legacy_display_name": legacy_name,
                    "legacy_score": parsed.get("score") if isinstance(parsed, dict) else None,
                    "legacy_details": parsed.get("details") if isinstance(parsed, dict) else None,
                    "verification_status": "legacy_unverified",
                    "warning": (
                        "legacy CGT did not invoke HumanEval check(candidate); not functional evidence"
                        if legacy_name in {"Code Generation Tests", "Code Completion Tests"}
                        else "recovered legacy value; not relabeled as revised evaluator output"
                    ),
                }
            except LegacyParseError as exc:
                diagnostics.append({"record_index": index, "field": legacy_name, "error": str(exc)})
        response = None
        raw_response = record.get("llm_response")
        if isinstance(raw_response, str):
            try:
                response = parse_legacy_repr(raw_response)
            except LegacyParseError as exc:
                diagnostics.append({"record_index": index, "field": "llm_response", "error": str(exc)})
        new_record.update({
            "legacy_record_index": index,
            "legacy_dimensions": dimensions,
            "recovered_response": response,
            "source_record_hash": fingerprint(record),
            "evaluator_version": "legacy-b30f4ab-or-earlier",
            "revised_score": None,
        })
        migrated.append(new_record)
    payload = {
        "schema_version": "2.0-legacy-adapter",
        "source": str(source.resolve()),
        "source_hash": fingerprint(report),
        "historical_only": True,
        "warning": "Historical scores are preserved but are not comparable to revised-v1 scores.",
        "records": migrated,
        "diagnostics": diagnostics,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "legacy-migration.json", payload)
    return output

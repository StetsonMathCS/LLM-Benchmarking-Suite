"""Experiment/model configuration with explicit CLI precedence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from facets.scoring_profile import PROJECT_ROOT, load_profile


ALL_TASKS = ["bug_fixing", "code_generation", "code_review", "refactoring", "test_generation", "translation"]


def load_yaml(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"configuration file not found: {resolved}")
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    data["_config_path"] = str(resolved)
    return data


def load_model_registry(path: str | Path | None = None) -> dict:
    registry_path = Path(path).resolve() if path else PROJECT_ROOT / "config" / "models.yaml"
    return yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}


def merge_cli(config: dict, overrides: dict[str, Any]) -> dict:
    """CLI values override YAML only when explicitly supplied (not None)."""
    resolved = dict(config)
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def resolve_model(
    alias: str | None, provider: str | None, model: str | None, registry: dict
) -> dict:
    if alias and (provider or model):
        raise ValueError("use either a model alias or --provider/--model, not both")
    if alias:
        entry = registry.get("models", {}).get(alias)
        if entry is None:
            raise ValueError(f"unknown model alias {alias!r}; run 'facets models list'")
        return {"alias": alias, **entry}
    if not provider or not model:
        raise ValueError("provide a model alias or both --provider and --model")
    capabilities = registry.get("provider_capabilities", {}).get(provider)
    if capabilities is None:
        raise ValueError(f"unknown provider {provider!r}")
    return {
        "alias": None,
        "provider": provider,
        "model": model,
        "credential_env": capabilities.get("credential_env"),
        "availability": "explicit_provider_id",
    }


def validate_resolved(config: dict, model_spec: dict, registry: dict) -> list[dict]:
    issues: list[dict] = []
    tasks = config.get("tasks", ALL_TASKS)
    unknown = sorted(set(tasks) - set(ALL_TASKS))
    if unknown:
        issues.append({"severity": "error", "code": "unknown_tasks", "detail": unknown})
    if int(config.get("samples", 1)) < 1:
        issues.append({"severity": "error", "code": "invalid_samples", "detail": "samples must be >= 1"})
    if config.get("limit") is not None and int(config["limit"]) < 1:
        issues.append({"severity": "error", "code": "invalid_limit", "detail": "limit must be >= 1"})
    if config.get("max_tokens") is not None and int(config["max_tokens"]) < 1:
        issues.append({"severity": "error", "code": "invalid_max_tokens", "detail": "max_tokens must be >= 1"})
    if config.get("temperature") is not None and not 0 <= float(config["temperature"]) <= 2:
        issues.append({"severity": "error", "code": "invalid_temperature", "detail": "temperature must be in [0, 2]"})
    if int(config.get("generation_concurrency", 1)) < 1:
        issues.append({"severity": "error", "code": "generation_concurrency", "detail": "must be >= 1"})
    if int(config.get("evaluator_concurrency", 1)) != 1:
        issues.append({"severity": "error", "code": "evaluator_concurrency", "detail": "revised-v1 requires sequential isolated evaluation"})
    if config.get("language", "python") != "python":
        issues.append({"severity": "error", "code": "study_language", "detail": "revised-v1 final study is Python-focused"})
    try:
        load_profile(str(config.get("profile", "revised-v1")))
    except Exception as exc:
        issues.append({"severity": "error", "code": "profile", "detail": str(exc)})
    availability = model_spec.get("availability")
    if availability in {"unresolved", "verify_with_provider"}:
        issues.append({
            "severity": "error" if availability == "unresolved" else "warning",
            "code": "model_availability",
            "detail": model_spec.get("note") or f"{model_spec.get('model')} requires an explicit availability check",
        })
    credential_env = model_spec.get("credential_env")
    if credential_env and not os.environ.get(credential_env):
        issues.append({"severity": "error", "code": "missing_credential", "detail": credential_env})
    extras = config.get("provider_extras", {}) or {}
    supported = set(registry.get("provider_capabilities", {}).get(model_spec["provider"], {}).get("supported_options", []))
    unsupported = sorted(set(extras) - supported)
    if unsupported:
        issues.append({"severity": "error", "code": "unsupported_provider_options", "detail": unsupported})
    return issues


def nonsecret_config(config: dict, model_spec: dict) -> dict:
    safe = {key: value for key, value in config.items() if key not in {"api_key"}}
    safe["model"] = {key: value for key, value in model_spec.items() if key != "api_key"}
    return safe


def fingerprint(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

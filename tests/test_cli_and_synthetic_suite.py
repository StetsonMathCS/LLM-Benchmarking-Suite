import json
from pathlib import Path
import subprocess
import sys

import pytest

from core.base import BaseProvider, BenchmarkResult, BenchmarkStatus, LLMResponse, ModelConfig
from datasets.mapper import DatasetRecord
from facets.cli import main
from facets.runner import execute_run


def test_offline_dry_run_never_constructs_provider(tmp_path, monkeypatch):
    config = tmp_path / "study.yaml"
    config.write_text("""\
profile: revised-v1
language: python
tasks: [code_generation]
models: [local-fake]
samples: 1
mutation_preflight: skip
execution: {backend: local}
""")
    registry = tmp_path / "models.yaml"
    registry.write_text("""\
models:
  local-fake: {provider: ollama, model: fake:latest, availability: explicit_provider_id}
provider_capabilities:
  ollama: {credential_env: null, supported_options: [temperature, max_tokens]}
""")
    monkeypatch.setattr("core.registry.ProviderRegistry.create", lambda config: (_ for _ in ()).throw(AssertionError("provider constructed")))
    assert main(["run", "local-fake", "--config", str(config), "--registry", str(registry), "--dry-run"]) == 0


def test_module_cli_help_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "facets.cli", "--help"], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "FACETS" in result.stdout


class FakeProvider(BaseProvider):
    def __init__(self):
        super().__init__(ModelConfig("ollama", "synthetic"))
        self.calls = 0
    def connect(self): return True
    def is_available(self): return True
    def complete(self, prompt, system_prompt=None):
        self.calls += 1
        return LLMResponse("synthetic response", "synthetic", "ollama")


class FakeBenchmark:
    name = "Synthetic Benchmark"
    def __init__(self, code_language, provider):
        self.provider = provider
    def build_prompt(self, language, code_input, **kwargs):
        return f"{language}:{code_input}"
    def _timed_run(self, code_input, **kwargs):
        response = self.provider.complete(self.build_prompt("python", code_input, **kwargs))
        return BenchmarkResult(self.name, BenchmarkStatus.PASSED, 1.0, llm_response=response)


def test_synthetic_six_task_orchestration(tmp_path, monkeypatch):
    tasks = ["bug_fixing", "code_generation", "code_review", "refactoring", "test_generation", "translation"]
    monkeypatch.setattr("datasets.mapper.DatasetMapper.load_dataset", lambda self, task, language, limit=None: [DatasetRecord(task, language, "1", {"buggy_code": "x", "prompt": "x", "code_snippet": "x", "original_code": "x"})])
    monkeypatch.setattr("datasets.mapper.DatasetMapper.get_task_class", lambda self, task: FakeBenchmark)
    monkeypatch.setattr("datasets.mapper.DatasetMapper.get_dataset_info", lambda self, task, language: {"task_name": task, "language": language, "csv_path": "synthetic", "record_count": 1, "record_ids_unique": True, "dataset_hash": task})
    provider = FakeProvider()
    config = {"profile": "revised-v1", "language": "python", "tasks": tasks, "samples": 1, "execution": {"backend": "local"}, "output_dir": str(tmp_path)}
    model = {"alias": "synthetic", "provider": "ollama", "model": "synthetic", "credential_env": None, "availability": "test"}
    run_dir, code = execute_run(config, model, tmp_path / "run", provider_override=provider)
    assert provider.calls == 6
    assert len((run_dir / "results.jsonl").read_text().splitlines()) == 6
    assert json.loads((run_dir / "manifest.json").read_text())["completed_count"] == 6


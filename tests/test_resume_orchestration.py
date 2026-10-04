import json

import pytest

from benchmarks.tasks.code_generation import CodeGenerationBenchmark
from core.base import BaseProvider, LLMResponse, ModelConfig
from facets.runner import execute_run


CORRECT_FIRST_HUMANEVAL = """\
from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for index, left in enumerate(numbers):
        for right in numbers[index + 1:]:
            if abs(left - right) < threshold:
                return True
    return False
"""


class CountingProvider(BaseProvider):
    def __init__(self, content):
        super().__init__(ModelConfig("ollama", "fake"))
        self.content = content
        self.calls = 0
    def connect(self): return True
    def is_available(self): return True
    def complete(self, prompt, system_prompt=None):
        self.calls += 1
        return LLMResponse(self.content, "fake", "ollama")


def config(tmp_path):
    return {
        "profile": "revised-v1", "language": "python", "translation_target": "javascript",
        "tasks": ["code_generation"], "samples": 1, "limit": 1,
        "output_dir": str(tmp_path), "execution": {"backend": "local"},
        "timeouts": {"evaluator_s": 2}, "retries": {"attempts": 1},
    }


def model():
    return {"alias": "fake", "provider": "ollama", "model": "fake", "credential_env": None, "availability": "test"}


def test_resume_reuses_response_after_evaluation_interruption(tmp_path, monkeypatch):
    provider = CountingProvider(CORRECT_FIRST_HUMANEVAL)
    run_dir = tmp_path / "run"
    original = CodeGenerationBenchmark._timed_run
    monkeypatch.setattr(CodeGenerationBenchmark, "_timed_run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupt after generation")))
    with pytest.raises(RuntimeError):
        execute_run(config(tmp_path), model(), run_dir, provider_override=provider)
    assert provider.calls == 1
    assert len(list((run_dir / "responses").glob("*.json"))) == 1

    monkeypatch.setattr(CodeGenerationBenchmark, "_timed_run", original)
    no_calls = CountingProvider("must not be used")
    path, _ = execute_run(config(tmp_path), model(), run_dir, resume=True, provider_override=no_calls)
    assert no_calls.calls == 0
    assert len((path / "results.jsonl").read_text().splitlines()) == 1


def test_completed_resume_does_not_duplicate_samples(tmp_path):
    provider = CountingProvider(CORRECT_FIRST_HUMANEVAL)
    run_dir = tmp_path / "run"
    execute_run(config(tmp_path), model(), run_dir, provider_override=provider)
    execute_run(config(tmp_path), model(), run_dir, resume=True, provider_override=CountingProvider("unused"))
    assert len((run_dir / "results.jsonl").read_text().splitlines()) == 1


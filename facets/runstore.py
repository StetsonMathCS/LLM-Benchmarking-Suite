"""Durable response-first run storage and atomic checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Iterator


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RunStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.responses = self.run_dir / "responses"
        self.results_path = self.run_dir / "results.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.checkpoint_path = self.run_dir / "checkpoint.json"

    def initialize(self, manifest: dict, resume: bool = False) -> None:
        if self.run_dir.exists() and not resume and any(self.run_dir.iterdir()):
            raise FileExistsError(f"run directory already exists and is nonempty: {self.run_dir}")
        self.responses.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != manifest.get("config_fingerprint"):
                raise ValueError("resume refused: configuration fingerprint differs from manifest")
        else:
            atomic_json(self.manifest_path, manifest)

    def response_path(self, identity: str) -> Path:
        return self.responses / f"{identity}.json"

    def save_response(self, identity: str, payload: dict) -> None:
        atomic_json(self.response_path(identity), payload)

    def load_response(self, identity: str) -> dict | None:
        path = self.response_path(identity)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def completed_identities(self) -> set[str]:
        completed: set[str] = set()
        if not self.results_path.exists():
            return completed
        with self.results_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # possible torn final line after abrupt power loss
                if record.get("record_identity"):
                    completed.add(record["record_identity"])
        return completed

    def append_result(self, payload: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def iter_results(self) -> Iterator[dict]:
        if not self.results_path.exists():
            return
        with self.results_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def checkpoint(self, payload: dict) -> None:
        atomic_json(self.checkpoint_path, payload)


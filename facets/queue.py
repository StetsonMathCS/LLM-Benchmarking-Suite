"""Durable sequential queue for multi-model experiments.

One queue owns one frozen experiment cohort. Items run strictly in order, every
state change is persisted before and after execution, and each item delegates to
``facets.runner.execute_run`` so scoring, evaluation, and token accounting stay in
exactly one place.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from facets.config import fingerprint, resolve_model, validate_resolved
from facets.runstore import atomic_json
from facets.usage import (
    LedgerScope,
    collect_run_ledgers,
    summarize_operations,
    summarize_usage,
    utc_now,
)

QUEUE_SCHEMA_VERSION = "1"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
STATUS_CANCELLED = "cancelled"
ALL_STATUSES = (
    STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED,
    STATUS_FAILED, STATUS_INTERRUPTED, STATUS_CANCELLED,
)

ON_ERROR_STOP = "stop"
ON_ERROR_CONTINUE = "continue"

# Fields that must match across items for their runs to form one comparable cohort.
COHORT_FIELDS = (
    "tasks", "language", "translation_target", "samples", "limit", "profile",
    "max_tokens", "temperature", "provider_extras",
)

EXIT_OK = 0
EXIT_INCOMPLETE = 3
EXIT_INTERRUPTED = 130
EXIT_CONFIG = 2


class QueueError(RuntimeError):
    """Queue state is unusable: lock contention, cohort drift, or bad request."""


def queue_root(output_root: str | Path) -> Path:
    """Queues live beside the run directories they drive, never inside a cohort."""
    return Path(output_root) / "queues"


def queue_dir(output_root: str | Path, name: str) -> Path:
    if not name or "/" in name or name in {".", ".."}:
        raise QueueError(f"invalid queue name {name!r}")
    return queue_root(output_root) / name


def _strip_private(config: dict) -> dict:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def cohort_of(config: dict) -> dict:
    """The subset of an experiment that must match for runs to be comparable."""
    return {field: config.get(field) for field in COHORT_FIELDS}


def cohort_differences(expected: dict, actual: dict) -> list[dict]:
    return [
        {"field": field, "expected": expected.get(field), "actual": actual.get(field)}
        for field in COHORT_FIELDS
        if expected.get(field) != actual.get(field)
    ]


@dataclass
class QueueItem:
    index: int
    alias: str
    model_spec: dict
    requested_alias: str | None = None
    status: str = STATUS_PENDING
    run_directory: str | None = None
    attempts: int = 0
    exit_code: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    issues: list | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "alias": self.alias,
            "requested_alias": self.requested_alias,
            "model": self.model_spec,
            "status": self.status,
            "run_directory": self.run_directory,
            "attempts": self.attempts,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "issues": self.issues,
        }

    @classmethod
    def from_dict(cls, value: dict) -> QueueItem:
        return cls(
            index=int(value["index"]),
            alias=str(value["alias"]),
            model_spec=value.get("model") or {},
            requested_alias=value.get("requested_alias"),
            status=value.get("status", STATUS_PENDING),
            run_directory=value.get("run_directory"),
            attempts=int(value.get("attempts", 0)),
            exit_code=value.get("exit_code"),
            started_at=value.get("started_at"),
            finished_at=value.get("finished_at"),
            error=value.get("error"),
            issues=value.get("issues"),
        )


class Queue:
    """Persistent, resumable, strictly sequential model queue."""

    def __init__(self, directory: Path, state: dict):
        self.directory = Path(directory)
        self.state = state

    # -- construction ----------------------------------------------------
    @classmethod
    def create(
        cls,
        name: str,
        config: dict,
        aliases: list[str],
        *,
        output_root: str | Path,
        registry: dict,
        on_error: str = ON_ERROR_STOP,
    ) -> Queue:
        if not aliases:
            raise QueueError("a queue needs at least one model alias")
        if on_error not in (ON_ERROR_STOP, ON_ERROR_CONTINUE):
            raise QueueError(f"on_error must be '{ON_ERROR_STOP}' or '{ON_ERROR_CONTINUE}'")
        output_root = Path(output_root)
        directory = queue_dir(output_root, name)
        if (directory / "queue.json").exists():
            raise QueueError(f"queue {name!r} already exists at {directory}")
        resolved_config = _strip_private({key: value for key, value in config.items() if key != "models"})
        items = []
        seen: dict[str, str] = {}
        for index, requested in enumerate(aliases):
            spec = resolve_model(requested, None, None, registry)
            alias = str(spec["alias"])
            if alias in seen:
                raise QueueError(
                    f"{requested!r} and {seen[alias]!r} both resolve to {alias!r}; "
                    "a queue runs each model once"
                )
            seen[alias] = requested
            items.append(QueueItem(
                index=index,
                alias=alias,
                model_spec=spec,
                requested_alias=requested if requested != alias else None,
                run_directory=str(Path(output_root) / name / alias),
            ))
        state = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "name": name,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": STATUS_PENDING,
            "on_error": on_error,
            "output_root": str(output_root),
            "run_root": str(Path(output_root) / name),
            "config_fingerprint": fingerprint(resolved_config),
            "cohort": cohort_of(resolved_config),
            "resolved_config": resolved_config,
            "registry_fingerprint": fingerprint({
                "models": registry.get("models", {}),
                "aliases": registry.get("aliases", {}),
            }),
            "items": [item.to_dict() for item in items],
            "finished_items": 0,
        }
        queue = cls(directory, state)
        directory.mkdir(parents=True, exist_ok=True)
        queue.save()
        queue.append_event("queue_created", name=name, items=[item.alias for item in items])
        return queue

    @classmethod
    def load(cls, name: str, output_root: str | Path) -> Queue:
        directory = queue_dir(output_root, name)
        path = directory / "queue.json"
        if not path.exists():
            raise QueueError(f"queue {name!r} not found at {directory}")
        state = json.loads(path.read_text(encoding="utf-8"))
        queue = cls(directory, state)
        recovered = queue.recover_stale_items()
        if recovered:
            queue.save()
        return queue

    # -- state -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.state["name"]

    @property
    def items(self) -> list[QueueItem]:
        return [QueueItem.from_dict(value) for value in self.state["items"]]

    @property
    def config(self) -> dict:
        return dict(self.state["resolved_config"])

    @property
    def cohort(self) -> dict:
        return dict(self.state["cohort"])

    @property
    def queue_id(self) -> str:
        return fingerprint({"name": self.name, "created_at": self.state["created_at"]})[:16]

    def status_counts(self) -> dict:
        counts = dict.fromkeys(ALL_STATUSES, 0)
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def recover_stale_items(self) -> list[str]:
        """A crashed process leaves items 'running'; they are resumable, not complete."""
        recovered = []
        for value in self.state["items"]:
            if value.get("status") == STATUS_RUNNING:
                value["status"] = STATUS_INTERRUPTED
                value["finished_at"] = value.get("finished_at") or utc_now()
                recovered.append(value["alias"])
        if recovered:
            self.append_event("stale_items_recovered", aliases=recovered)
        return recovered

    def derived_status(self) -> str:
        """Aggregate status from item states.

        Failures and interruptions outrank leftover pending work because they are
        what a reader must act on; counts still show what never ran.
        """
        counts = self.status_counts()
        if counts[STATUS_RUNNING]:
            return STATUS_RUNNING
        if counts[STATUS_INTERRUPTED]:
            return STATUS_INTERRUPTED
        if counts[STATUS_FAILED] or counts[STATUS_CANCELLED]:
            return STATUS_FAILED
        if counts[STATUS_PENDING]:
            return STATUS_PENDING
        return STATUS_COMPLETED

    def save(self) -> None:
        self.state["updated_at"] = utc_now()
        self.state["finished_items"] = sum(
            1 for item in self.items if item.status == STATUS_COMPLETED
        )
        self.state["status"] = self.derived_status()
        atomic_json(self.directory / "queue.json", self.state)

    def append_event(self, kind: str, **fields) -> None:
        entry = {"at": utc_now(), "event": kind, **fields}
        path = self.directory / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def events(self) -> list[dict]:
        path = self.directory / "events.jsonl"
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def _update_item(self, item: QueueItem, **changes) -> None:
        for value in self.state["items"]:
            if value["index"] == item.index:
                value.update(changes)
                return
        raise QueueError(f"queue item {item.index} is missing from the queue")

    # -- locking ---------------------------------------------------------
    def lock(self):
        return _QueueLock(self.directory)

    # -- selection -------------------------------------------------------
    def selectable(self, *, retry_failed: bool = False) -> list[QueueItem]:
        """Items still needing work, in queue order."""
        selected = []
        for item in self.items:
            if item.status == STATUS_COMPLETED:
                continue
            if item.status == STATUS_FAILED and not retry_failed:
                continue
            if item.status == STATUS_CANCELLED:
                continue
            selected.append(item)
        return selected

    def verify_registry(self, registry: dict) -> None:
        """Refuse to continue a queue whose frozen model identities changed."""
        current = fingerprint({
            "models": registry.get("models", {}),
            "aliases": registry.get("aliases", {}),
        })
        if current != self.state.get("registry_fingerprint"):
            raise QueueError(
                "registry changed since the queue was created; "
                "create a new queue or restore the frozen registry entries"
            )

    def verify_cohort(self, overrides: dict | None = None) -> None:
        """Refuse to mix incomparable runs inside one queue."""
        if not overrides:
            return
        merged = {**self.config, **{k: v for k, v in overrides.items() if v is not None}}
        differences = cohort_differences(self.cohort, cohort_of(merged))
        if differences:
            raise QueueError(
                "queue cohort mismatch: " + json.dumps(differences, sort_keys=True)
            )

    # -- execution -------------------------------------------------------
    def execute(
        self,
        *,
        registry: dict,
        on_error: str | None = None,
        retry_failed: bool = False,
        dry_run: bool = False,
        on_event: Callable[[dict], None] | None = None,
        runner: Callable[..., tuple] | None = None,
    ) -> dict:
        """Run remaining items in order and return the queue summary."""
        from facets.runner import execute_run

        runner = runner or execute_run
        policy = on_error or self.state.get("on_error") or ON_ERROR_STOP
        if policy not in (ON_ERROR_STOP, ON_ERROR_CONTINUE):
            raise QueueError(f"on_error must be '{ON_ERROR_STOP}' or '{ON_ERROR_CONTINUE}'")
        self.verify_registry(registry)
        plan = self.selectable(retry_failed=retry_failed)
        if dry_run:
            return self.summary(planned=[item.alias for item in plan], dry_run=True)

        self.state["on_error"] = policy
        self.save()
        self.append_event("queue_started", policy=policy, planned=[item.alias for item in plan])

        stopped_early = False
        for item in plan:
            issues = validate_resolved(self.config, item.model_spec, registry)
            if any(issue["severity"] == "error" for issue in issues):
                self._fail_item(item, STATUS_FAILED, "preflight_failed", error="preflight rejected the item", issues=issues)
                if policy == ON_ERROR_STOP:
                    stopped_early = True
                    break
                continue
            existing = Path(item.run_directory)
            resume = (existing / "manifest.json").exists()
            self._update_item(
                item,
                status=STATUS_RUNNING,
                started_at=utc_now(),
                attempts=item.attempts + 1,
                error=None,
                issues=None,
            )
            self.save()
            self.append_event("item_started", alias=item.alias, resume=resume, attempt=item.attempts + 1)
            if on_event:
                on_event({"event": "item_started", "alias": item.alias, "resume": resume})
            scope = LedgerScope(
                operation_id=self.queue_id,
                queue_id=self.name,
                queue_item_id=item.alias,
                run_id=item.alias,
            )
            try:
                _, code = runner(
                    self.config,
                    item.model_spec,
                    existing,
                    resume=resume,
                    usage_scope=scope,
                )
            except KeyboardInterrupt:
                self._fail_item(item, STATUS_INTERRUPTED, "interrupted", error="keyboard interrupt", exit_code=EXIT_INTERRUPTED)
                self.append_event("queue_interrupted", alias=item.alias)
                self.save()
                summary = self.summary(stopped_early=True)
                self._write_summary(summary)
                return summary
            if code == EXIT_OK:
                self._update_item(
                    item,
                    status=STATUS_COMPLETED,
                    exit_code=code,
                    finished_at=utc_now(),
                    run_directory=str(existing),
                )
                self.append_event("item_completed", alias=item.alias, run_directory=str(existing))
                if on_event:
                    on_event({"event": "item_completed", "alias": item.alias})
            else:
                self._fail_item(item, STATUS_FAILED, "run_failed", error=f"execute_run exit code {code}", exit_code=code)
                if policy == ON_ERROR_STOP:
                    stopped_early = True
                    break
            self.save()

        summary = self.summary(stopped_early=stopped_early)
        self.save()
        self._write_summary(summary)
        self.append_event("queue_finished", status=summary["status"], stopped_early=stopped_early)
        return summary

    def _fail_item(self, item: QueueItem, status: str, event: str, **fields) -> None:
        self._update_item(item, status=status, finished_at=utc_now(), **fields)
        self.save()
        self.append_event(event, alias=item.alias, **{
            key: value for key, value in fields.items() if key in {"error", "exit_code", "issues"}
        })

    # -- reporting -------------------------------------------------------
    def usage_entries(self) -> list[dict]:
        return collect_run_ledgers(Path(self.state["run_root"]))

    def summary(self, *, planned: list[str] | None = None, dry_run: bool = False,
                stopped_early: bool = False) -> dict:
        entries = self.usage_entries()
        usage = summarize_usage(entries)
        counts = self.status_counts()
        items = [item.to_dict() for item in self.items]
        incomplete = [
            item["alias"] for item in items
            if item["status"] not in (STATUS_COMPLETED,)
        ]
        summary = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "name": self.name,
            "status": self.derived_status(),
            "generated_at": utc_now(),
            "dry_run": dry_run,
            "stopped_early": stopped_early,
            "on_error": self.state.get("on_error"),
            "cohort": self.cohort,
            "config_fingerprint": self.state["config_fingerprint"],
            "run_root": self.state["run_root"],
            "item_counts": counts,
            "items": items,
            "planned": planned,
            "incomplete_items": incomplete,
            "usage": usage,
            "usage_by_item": summarize_operations(entries),
        }
        if not dry_run:
            summary["exit_code"] = _queue_exit_code(summary)
        else:
            summary["exit_code"] = EXIT_OK
        return summary

    def _write_summary(self, summary: dict) -> None:
        atomic_json(self.directory / "summary.json", summary)

    def read_summary(self) -> dict:
        path = self.directory / "summary.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return self.summary()


def _queue_exit_code(summary: dict) -> int:
    if summary["status"] == STATUS_INTERRUPTED:
        return EXIT_INTERRUPTED
    if summary["status"] == STATUS_COMPLETED:
        return EXIT_OK
    return EXIT_INCOMPLETE


class _QueueLock:
    """Exclusive OS-level lock so two processes cannot drive one queue."""

    def __init__(self, directory: Path):
        self.path = Path(directory) / "queue.lock"
        self.handle = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise QueueError(f"queue is already being driven by another process ({self.path})") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "at": utc_now()}) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, *exc_info) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def list_queues(output_root: str | Path) -> list[dict]:
    root = queue_root(output_root)
    if not root.exists():
        return []
    rows = []
    for directory in sorted(root.iterdir()):
        path = directory / "queue.json"
        if not path.exists():
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        counts = dict.fromkeys(ALL_STATUSES, 0)
        for item in state.get("items", []):
            counts[item.get("status", STATUS_PENDING)] = counts.get(item.get("status", STATUS_PENDING), 0) + 1
        rows.append({
            "name": state.get("name"),
            "status": state.get("status"),
            "updated_at": state.get("updated_at"),
            "created_at": state.get("created_at"),
            "cohort": state.get("cohort"),
            "item_counts": counts,
            "run_root": state.get("run_root"),
        })
    return rows


def queue_name_for_config(config_path: str | Path, override: str | None = None) -> str:
    if override:
        return override
    return f"sweep-{Path(config_path).stem}"


__all__ = [
    "COHORT_FIELDS",
    "ON_ERROR_CONTINUE",
    "ON_ERROR_STOP",
    "STATUS_CANCELLED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_INTERRUPTED",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "Queue",
    "QueueError",
    "QueueItem",
    "cohort_differences",
    "cohort_of",
    "list_queues",
    "queue_dir",
    "queue_name_for_config",
    "queue_root",
]

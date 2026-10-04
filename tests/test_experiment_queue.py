import json

import pytest

from facets.queue import (
    ON_ERROR_CONTINUE,
    Queue,
    QueueError,
    cohort_differences,
    cohort_of,
    list_queues,
)


def _config(tmp_path, **overrides):
    config = {
        "profile": "revised-v1",
        "language": "python",
        "translation_target": "javascript",
        "tasks": ["code_generation", "bug_fixing"],
        "samples": 1,
        "limit": 1,
        "output_dir": str(tmp_path / "runs"),
        "execution": {"backend": "local"},
        "timeouts": {"evaluator_s": 2},
        "retries": {"attempts": 1},
        "models": ["alpha", "beta", "gamma"],
    }
    config.update(overrides)
    return config


def _registry():
    return {
        "models": {
            alias: {"provider": "ollama", "model": f"{alias}-model", "credential_env": None,
                    "availability": "test"}
            for alias in ("alpha", "beta", "gamma")
        },
        "provider_capabilities": {
            "ollama": {"credential_env": None, "supported_options": []},
        },
    }


def _create(tmp_path, **kwargs):
    config = _config(tmp_path, **kwargs.pop("config", {}))
    return Queue.create(
        kwargs.pop("name", "study"),
        config,
        kwargs.pop("aliases", ["alpha", "beta", "gamma"]),
        output_root=tmp_path / "runs",
        registry=_registry(),
        **kwargs,
    )


def _fake_runner(codes=None, calls=None):
    codes = codes or {}
    calls = calls if calls is not None else []

    def runner(config, model_spec, run_dir, *, resume=False, usage_scope=None):
        calls.append({"alias": model_spec.get("alias"), "model": model_spec.get("model"),
                      "run_dir": str(run_dir), "resume": resume})
        (run_dir / "usage").mkdir(parents=True, exist_ok=True)
        (run_dir / "usage" / "ledger.jsonl").write_text(json.dumps({
            "request_id": f"{model_spec['alias']}-1",
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "purpose": "generation",
            "provider": model_spec["provider"],
            "model": model_spec["model"],
            "queue_id": usage_scope.queue_id if usage_scope else None,
            "queue_item_id": usage_scope.queue_item_id if usage_scope else None,
            "response_status": "success",
            "usage_origin": "measured_here",
            "usage": {"measured": True, "counted_input_tokens": 100, "counted_output_tokens": 20},
            "counted_total_tokens": 120,
        }) + "\n")
        code = codes.get(model_spec.get("alias"), 0)
        (run_dir / "manifest.json").write_text("{}")
        return run_dir, code

    return runner


def test_queue_persists_items_in_order(tmp_path):
    queue = _create(tmp_path)
    assert [item.alias for item in queue.items] == ["alpha", "beta", "gamma"]
    assert [item.status for item in queue.items] == ["pending"] * 3
    assert queue.state["cohort"]["tasks"] == ["code_generation", "bug_fixing"]
    assert (queue.directory / "queue.json").exists()
    assert queue.events()[0]["event"] == "queue_created"


def test_queue_is_exclusive_on_create(tmp_path):
    _create(tmp_path)
    with pytest.raises(QueueError):
        _create(tmp_path)


def test_unknown_alias_is_rejected_at_creation(tmp_path):
    with pytest.raises(ValueError):
        _create(tmp_path, aliases=["alpha", "nonesuch"])


def test_queue_runs_items_sequentially_and_stops_on_failure(tmp_path):
    queue = _create(tmp_path)
    calls = []
    summary = queue.execute(
        registry=_registry(),
        runner=_fake_runner(codes={"beta": 3}, calls=calls),
    )
    assert [call["alias"] for call in calls] == ["alpha", "beta"]
    assert [item.status for item in queue.items] == ["completed", "failed", "pending"]
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 3
    assert summary["stopped_early"] is True
    assert summary["incomplete_items"] == ["beta", "gamma"]


def test_queue_continue_policy_runs_every_item(tmp_path):
    queue = _create(tmp_path, on_error=ON_ERROR_CONTINUE)
    calls = []
    summary = queue.execute(
        registry=_registry(),
        runner=_fake_runner(codes={"alpha": 3, "beta": 3}, calls=calls),
    )
    assert [call["alias"] for call in calls] == ["alpha", "beta", "gamma"]
    assert [item.status for item in queue.items] == ["failed", "failed", "completed"]
    assert summary["item_counts"]["failed"] == 2
    assert summary["exit_code"] == 3


def test_queue_reports_usage_per_item(tmp_path):
    queue = _create(tmp_path)
    summary = queue.execute(registry=_registry(), runner=_fake_runner())
    assert summary["usage"]["counted_total_tokens"] == 360
    assert summary["usage"]["by_purpose"]["generation"]["requests_total"] == 3
    assert summary["usage_by_item"]["by_queue_item"]["alpha"]["counted_total_tokens"] == 120
    assert summary["usage_by_item"]["by_queue_item"]["gamma"]["counted_total_tokens"] == 120


def test_queue_resume_skips_completed_and_retries_failed_only_when_asked(tmp_path):
    queue = _create(tmp_path)
    queue.execute(registry=_registry(), runner=_fake_runner(codes={"beta": 3}))

    calls = []
    queue.execute(registry=_registry(), runner=_fake_runner(codes={}, calls=calls))
    assert [call["alias"] for call in calls] == ["gamma"]
    assert [call["resume"] for call in calls] == [False]

    calls.clear()
    queue.execute(registry=_registry(), runner=_fake_runner(codes={}, calls=calls), retry_failed=True)
    assert [call["alias"] for call in calls] == ["beta"]
    assert all(call["resume"] for call in calls)
    assert [item.status for item in queue.items] == ["completed"] * 3


def test_keyboard_interrupt_marks_item_and_queue(tmp_path):
    queue = _create(tmp_path)

    def runner(config, model_spec, run_dir, *, resume=False, usage_scope=None):
        raise KeyboardInterrupt

    summary = queue.execute(registry=_registry(), runner=runner)
    assert queue.items[0].status == "interrupted"
    assert summary["status"] == "interrupted"
    assert summary["exit_code"] == 130


def test_preflight_failure_is_recorded_as_failed_item(tmp_path):
    registry = _registry()
    registry["models"]["alpha"]["availability"] = "unresolved"
    queue = Queue.create("study", _config(tmp_path), ["alpha", "beta"], output_root=tmp_path / "runs",
                         registry=registry, on_error=ON_ERROR_CONTINUE)
    calls = []
    summary = queue.execute(registry=registry, runner=_fake_runner(calls=calls))
    assert [call["alias"] for call in calls] == ["beta"]
    assert queue.items[0].status == "failed"
    assert any(issue["code"] == "model_availability" for issue in queue.items[0].issues)
    assert summary["exit_code"] == 3


def test_dry_run_executes_nothing(tmp_path):
    queue = _create(tmp_path)
    calls = []
    summary = queue.execute(registry=_registry(), runner=_fake_runner(calls=calls), dry_run=True)
    assert calls == []
    assert summary["planned"] == ["alpha", "beta", "gamma"]
    assert summary["dry_run"] is True
    assert summary["exit_code"] == 0
    assert not (queue.directory / "summary.json").exists()


def test_queue_lock_blocks_concurrent_driver(tmp_path):
    queue = _create(tmp_path)
    with queue.lock(), pytest.raises(QueueError), queue.lock():
        pass
    with queue.lock():
        pass


def test_events_record_lifecycle(tmp_path):
    queue = _create(tmp_path)
    queue.execute(registry=_registry(), runner=_fake_runner())
    kinds = [event["event"] for event in queue.events()]
    assert kinds == [
        "queue_created", "queue_started", "item_started", "item_completed",
        "item_started", "item_completed", "item_started", "item_completed", "queue_finished",
    ]


def test_stale_running_items_are_recovered_on_load(tmp_path):
    queue = _create(tmp_path)
    queue._update_item(queue.items[1], status="running", started_at="2026-01-01T00:00:00+00:00")
    queue.save()
    reloaded = Queue.load("study", tmp_path / "runs")
    assert reloaded.items[1].status == "interrupted"
    assert [event["event"] for event in reloaded.events()][-1] == "stale_items_recovered"


def test_summary_is_written_after_execution(tmp_path):
    queue = _create(tmp_path)
    queue.execute(registry=_registry(), runner=_fake_runner())
    stored = json.loads((queue.directory / "summary.json").read_text())
    assert stored["status"] == "completed"
    assert stored["exit_code"] == 0
    assert queue.read_summary()["item_counts"]["completed"] == 3


def test_list_queues_reports_progress(tmp_path):
    _create(tmp_path)
    queue = Queue.load("study", tmp_path / "runs")
    queue.execute(registry=_registry(), runner=_fake_runner(codes={"beta": 3}), )
    rows = list_queues(tmp_path / "runs")
    assert len(rows) == 1
    assert rows[0]["name"] == "study"
    assert rows[0]["item_counts"]["failed"] == 1
    assert rows[0]["item_counts"]["pending"] == 1


def test_cohort_mismatch_is_refused(tmp_path):
    queue = _create(tmp_path)
    queue.verify_cohort({"limit": 1})
    with pytest.raises(QueueError):
        queue.verify_cohort({"limit": 5})
    differences = cohort_differences(queue.cohort, cohort_of(_config(tmp_path, samples=3)))
    assert differences == [{"field": "samples", "expected": 1, "actual": 3}]


def test_queue_rejects_invalid_name(tmp_path):
    with pytest.raises(QueueError):
        Queue.create("../escape", _config(tmp_path), ["alpha"], output_root=tmp_path / "runs",
                     registry=_registry())


def test_queue_aliases_resolve_to_canonical_model_identity(tmp_path):
    registry = _registry()
    registry["aliases"] = {"one": "alpha"}
    queue = Queue.create("study", _config(tmp_path), ["one", "beta"], output_root=tmp_path / "runs",
                         registry=registry)
    assert [item.alias for item in queue.items] == ["alpha", "beta"]
    assert queue.items[0].requested_alias == "one"
    assert queue.items[1].requested_alias is None
    assert queue.items[0].run_directory.endswith("study/alpha")


def test_queue_refuses_two_spellings_of_one_model(tmp_path):
    registry = _registry()
    registry["aliases"] = {"one": "alpha", "uno": "alpha"}
    with pytest.raises(QueueError, match="a queue runs each model once"):
        Queue.create("study", _config(tmp_path), ["one", "uno"], output_root=tmp_path / "runs",
                     registry=registry)


def test_frozen_registry_is_enforced(tmp_path):
    queue = _create(tmp_path)
    queue.verify_registry(_registry())

    renamed = _registry()
    renamed["models"]["beta"]["model"] = "beta-v2"
    with pytest.raises(QueueError, match="registry changed"):
        queue.verify_registry(renamed)

    realiased = _registry()
    realiased["aliases"] = {"two": "beta"}
    with pytest.raises(QueueError, match="registry changed"):
        queue.verify_registry(realiased)


def test_missing_queue_is_reported(tmp_path):
    with pytest.raises(QueueError):
        Queue.load("nonesuch", tmp_path / "runs")

import json

from facets.cli import main
from facets.queue import Queue


def _write_config(tmp_path, aliases, **extra):
    config = tmp_path / "study.yaml"
    lines = [
        "profile: revised-v1",
        "language: python",
        "tasks: [code_generation]",
        f"models: [{', '.join(aliases)}]",
        "samples: 1",
        "limit: 1",
        "mutation_preflight: skip",
        "execution: {backend: local}",
        f"output_dir: {tmp_path / 'runs'}",
    ]
    lines += [f"{key}: {value}" for key, value in extra.items()]
    config.write_text("\n".join(lines) + "\n")
    return config


def _write_registry(tmp_path, aliases, name="models.yaml"):
    registry = tmp_path / name
    entries = "\n".join(
        f"  {alias}: {{provider: ollama, model: {alias}-model, availability: explicit_provider_id}}"
        for alias in aliases
    )
    registry.write_text(f"""\
schema_version: 1
aliases: {{}}
models:
{entries}
provider_capabilities:
  ollama: {{credential_env: null, supported_options: [temperature, max_tokens]}}
""")
    return registry


def _fake_runner(codes=None, calls=None):
    codes = codes or {}
    calls = calls if calls is not None else []

    def runner(config, model_spec, run_dir, *, resume=False, usage_scope=None):
        calls.append(model_spec.get("alias"))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({"model": model_spec["model"]}))
        return run_dir, codes.get(model_spec.get("alias"), 0)

    return runner, calls


def test_queue_create_and_list(tmp_path, capsys):
    config = _write_config(tmp_path, ["alpha", "beta"])
    registry = _write_registry(tmp_path, ["alpha", "beta"])

    code = main([
        "queue", "create", "study", "--config", str(config), "--registry", str(registry),
        "--output-dir", str(tmp_path / "runs"),
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["planned"] == ["alpha", "beta"]
    assert payload["status"] == "pending"

    assert main(["queue", "list", "--output-dir", str(tmp_path / "runs")]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in rows] == ["study"]
    assert rows[0]["item_counts"]["pending"] == 2


def test_queue_create_refuses_duplicate_name(tmp_path, capsys):
    config = _write_config(tmp_path, ["alpha"])
    registry = _write_registry(tmp_path, ["alpha"])
    args = ["queue", "create", "study", "--config", str(config), "--registry", str(registry),
            "--output-dir", str(tmp_path / "runs")]
    assert main(args) == 0
    capsys.readouterr()
    assert main(args) == 2
    assert "already exists" in capsys.readouterr().err


def test_queue_run_status_and_resume(tmp_path, capsys, monkeypatch):
    config = _write_config(tmp_path, ["alpha", "beta"])
    registry = _write_registry(tmp_path, ["alpha", "beta"])
    failing, first_calls = _fake_runner(codes={"beta": 3})
    monkeypatch.setattr("facets.runner.execute_run", failing)

    main(["queue", "create", "study", "--config", str(config), "--registry", str(registry),
          "--output-dir", str(tmp_path / "runs")])
    capsys.readouterr()

    assert main(["queue", "run", "study", "--output-dir", str(tmp_path / "runs"),
                 "--registry", str(registry)]) == 3
    capsys.readouterr()
    assert first_calls == ["alpha", "beta"]

    assert main(["queue", "status", "study", "--output-dir", str(tmp_path / "runs")]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "failed"
    assert status["item_counts"]["completed"] == 1
    assert status["exit_code"] == 3

    healthy, retry_calls = _fake_runner()
    monkeypatch.setattr("facets.runner.execute_run", healthy)
    assert main(["queue", "run", "study", "--output-dir", str(tmp_path / "runs"),
                 "--registry", str(registry), "--retry-failed"]) == 0
    capsys.readouterr()
    assert retry_calls == ["beta"]
    assert Queue.load("study", tmp_path / "runs").read_summary()["exit_code"] == 0


def test_queue_dry_run_executes_nothing(tmp_path, capsys, monkeypatch):
    config = _write_config(tmp_path, ["alpha", "beta"])
    registry = _write_registry(tmp_path, ["alpha", "beta"])
    runner, calls = _fake_runner()
    monkeypatch.setattr("facets.runner.execute_run", runner)

    main(["queue", "create", "study", "--config", str(config), "--registry", str(registry),
          "--output-dir", str(tmp_path / "runs")])
    capsys.readouterr()
    assert main(["queue", "run", "study", "--dry-run", "--output-dir", str(tmp_path / "runs"),
                 "--registry", str(registry)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["planned"] == ["alpha", "beta"]
    assert calls == []


def test_queue_run_refuses_a_different_registry(tmp_path, capsys):
    config = _write_config(tmp_path, ["alpha"])
    registry = _write_registry(tmp_path, ["alpha"])
    other = _write_registry(tmp_path, ["alpha", "beta"], name="other")
    main(["queue", "create", "study", "--config", str(config), "--registry", str(registry),
          "--output-dir", str(tmp_path / "runs")])
    capsys.readouterr()

    assert main(["queue", "run", "study", "--output-dir", str(tmp_path / "runs"),
                 "--registry", str(other)]) == 2
    assert "registry changed" in capsys.readouterr().err


def test_queue_run_on_missing_queue_reports_error(tmp_path, capsys):
    assert main(["queue", "run", "nonesuch", "--output-dir", str(tmp_path / "runs")]) == 2
    assert "not found" in capsys.readouterr().err


def test_sweep_creates_and_drives_a_queue(tmp_path, capsys, monkeypatch):
    config = _write_config(tmp_path, ["alpha", "beta"])
    registry = _write_registry(tmp_path, ["alpha", "beta"])
    runner, calls = _fake_runner()
    monkeypatch.setattr("facets.runner.execute_run", runner)

    assert main(["sweep", "--config", str(config), "--registry", str(registry)]) == 0
    capsys.readouterr()
    assert calls == ["alpha", "beta"]

    queue = Queue.load("sweep-study", tmp_path / "runs")
    assert [item.status for item in queue.items] == ["completed", "completed"]
    assert queue.summary()["exit_code"] == 0


def test_sweep_reports_failure_with_nonzero_exit(tmp_path, capsys, monkeypatch):
    config = _write_config(tmp_path, ["alpha", "beta"])
    registry = _write_registry(tmp_path, ["alpha", "beta"])
    runner, calls = _fake_runner(codes={"alpha": 3})
    monkeypatch.setattr("facets.runner.execute_run", runner)

    assert main(["sweep", "--config", str(config), "--registry", str(registry),
                 "--on-error", "continue"]) == 3
    capsys.readouterr()
    summary = Queue.load("sweep-study", tmp_path / "runs").read_summary()
    assert summary["item_counts"]["failed"] == 1
    assert summary["item_counts"]["completed"] == 1
    assert calls == ["alpha", "beta"]


def test_sweep_resume_refuses_changed_cohort(tmp_path, capsys, monkeypatch):
    config = _write_config(tmp_path, ["alpha"])
    registry = _write_registry(tmp_path, ["alpha"])
    runner, _ = _fake_runner()
    monkeypatch.setattr("facets.runner.execute_run", runner)
    main(["sweep", "--config", str(config), "--registry", str(registry)])
    capsys.readouterr()

    config.write_text(config.read_text().replace("samples: 1", "samples: 2"))
    assert main(["sweep", "--config", str(config), "--registry", str(registry), "--resume"]) == 2
    assert "cohort" in capsys.readouterr().err


def test_sweep_without_models_is_rejected(tmp_path, capsys):
    config = tmp_path / "empty.yaml"
    config.write_text("profile: revised-v1\nmodels: []\n")
    assert main(["sweep", "--config", str(config)]) == 2
    assert "no models" in capsys.readouterr().err

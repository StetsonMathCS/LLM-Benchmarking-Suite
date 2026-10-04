"""Offline figure and run-selection behaviour, verified from synthetic runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from facets.analysis import analyze_directory, discover_runs, select_runs
from facets.cli import main
from facets.figures import FIGURE_FORMATS

TASKS = ("code_generation", "code_review", "bug_fixing")
DATASET_HASHES = {task: f"hash-{task}" for task in TASKS}


def _write_run(
    root: Path,
    name: str,
    model: str,
    scores: dict[str, float] | None = None,
    *,
    status: str = "complete",
    parameter_size: str | None = None,
    completed_at: str | None = None,
    profile_hash: str = "profile-hash",
    evaluator: str = "evaluator-v1",
    dataset_hashes: dict[str, str] | None = None,
    latency_ms: float | None = None,
    records_per_task: int = 6,
) -> Path:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1",
        "status": status,
        "model_alias": model,
        "provider": "fixture",
        "model": f"{model}-tag",
        "evaluator_version": evaluator,
        "scoring_profile": "revised-v1",
        "scoring_profile_hash": profile_hash,
        "resolved_config": {"model": {"parameter_size": parameter_size}} if parameter_size else {"model": {}},
        "dataset": {task: {"dataset_hash": value} for task, value in (dataset_hashes or DATASET_HASHES).items()},
        "planned_count": len(TASKS) * records_per_task,
        "generated_at": "2026-04-01T00:00:00+00:00",
        "completed_at": completed_at,
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    records = []
    for task_index, task in enumerate(TASKS):
        for index in range(records_per_task):
            score = (scores or {}).get(task, 0.4 + 0.1 * task_index + 0.01 * index)
            records.append({
                "task": task,
                "record_id": f"{task}-{index}",
                "sample_index": 0,
                "record_identity": f"{name}:{task}:{index}",
                "combined_score": score if status == "complete" or index < 2 else None,
                "status": "complete" if score is not None else "failed",
                "metadata": {"functional_correct": index % 2 == 0},
                "details": {
                    "runtime_analysis": {
                        "display_name": "Runtime",
                        "score": score / 2,
                        "applicable": True,
                        "status": "complete",
                        "details": {"raw_time_ratio": 1.1 + index / 10, "raw_memory_ratio": 1.2},
                    }
                },
                "llm_response": {
                    "latency_ms": (latency_ms or 800) + 40 * index + 100 * task_index,
                    "usage": {"input_tokens": 1000, "output_tokens": 200},
                },
            })
    (run / "results.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return run


def _figure_files(output: Path) -> list[Path]:
    wanted = {f".{extension}" for extension in FIGURE_FORMATS}
    return sorted(path for path in output.rglob("*") if path.suffix in wanted)


def test_analysis_renders_every_publication_figure(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-run", "alpha", {"code_generation": 0.8, "code_review": 0.6, "bug_fixing": 0.4},
               parameter_size="7B", latency_ms=700)
    _write_run(runs, "beta-run", "beta", {"code_generation": 0.5, "code_review": 0.7, "bug_fixing": 0.6},
               parameter_size="3B", latency_ms=1500)

    summary = analyze_directory(runs, tmp_path / "analysis")

    assert summary["warnings"] == []
    assert summary["chart_status"] == "generated"
    assert set(summary["figures"]) == {
        "record_score_distributions",
        "task_capability_radar",
        "model_task_score_heatmap",
        "task_score_grouped_bars",
        "score_vs_parameter_size",
        "score_vs_generation_latency",
    }
    files = _figure_files(tmp_path / "analysis")
    assert {path.suffix for path in files} == {".png", ".pdf", ".svg"}
    for stem in summary["figures"]:
        for extension in FIGURE_FORMATS:
            assert (tmp_path / "analysis" / f"{stem}.{extension}").stat().st_size > 0

    index = (tmp_path / "analysis" / "index.md").read_text(encoding="utf-8")
    for stem in summary["figures"]:
        assert stem in index
    assert "![record_score_distributions](record_score_distributions.png)" in index

    manifest = json.loads((tmp_path / "analysis" / "figures.json").read_text(encoding="utf-8"))
    assert manifest["formats"] == list(FIGURE_FORMATS)
    assert manifest["failures"] == []
    captions = {entry["stem"]: entry["caption"] for entry in manifest["figures"]}
    assert "must not be read as a weighted composite" in captions["task_capability_radar"]
    assert "excluded rather than assumed" in captions["score_vs_parameter_size"]
    for entry in manifest["figures"]:
        assert entry["files"]["png"].endswith(".png")
        assert set(entry["files"]) == set(FIGURE_FORMATS)


def test_analysis_exports_tidy_tables_for_every_figure(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-run", "alpha", parameter_size="7B", latency_ms=700)
    _write_run(runs, "beta-run", "beta", parameter_size="3B", latency_ms=1500)

    analyze_directory(runs, tmp_path / "analysis")

    output = tmp_path / "analysis"
    for table in ("records.csv", "dimensions.csv", "runtime_ratios.csv", "task_summary.csv",
                  "dimension_summary.csv", "parameter_sizes.csv", "generation_latency.csv",
                  "sensitivity.csv", "paired_bootstrap.csv"):
        assert (output / table).exists(), table
    latency_rows = (output / "generation_latency.csv").read_text(encoding="utf-8").splitlines()
    assert latency_rows[0].split(",") == ["model", "task", "record_id", "latency_ms"]
    assert len(latency_rows) == 1 + 2 * len(TASKS) * 6
    records = (output / "records.csv").read_text(encoding="utf-8").splitlines()
    header = records[0].split(",")
    assert "latency_ms" in header and "input_tokens" in header


def test_undisclosed_parameter_size_is_excluded_not_assumed(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-run", "alpha", parameter_size="7B")
    _write_run(runs, "beta-run", "beta")
    _write_run(runs, "gamma-run", "gamma", parameter_size="70B")

    summary = analyze_directory(runs, tmp_path / "analysis")

    assert summary["undisclosed_or_unverified_parameter_sizes"] == ["beta"]
    assert set(summary["known_parameter_sizes_billions"]) == {"alpha", "gamma"}
    manifest = json.loads((tmp_path / "analysis" / "figures.json").read_text(encoding="utf-8"))
    scatter = next(entry for entry in manifest["figures"] if entry["stem"] == "score_vs_parameter_size")
    assert scatter["excluded"] == ["beta"]
    assert scatter["models"] == ["alpha", "gamma"]


def test_repeated_models_are_ambiguous_by_default(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-old", "alpha", completed_at="2026-04-01T00:00:00+00:00")
    _write_run(runs, "alpha-new", "alpha", completed_at="2026-05-01T00:00:00+00:00")
    _write_run(runs, "beta-run", "beta")

    with pytest.raises(ValueError, match="ambiguous repeated models"):
        analyze_directory(runs, tmp_path / "analysis")


def test_latest_complete_selects_newest_run_per_model(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-old", "alpha", completed_at="2026-04-01T00:00:00+00:00")
    _write_run(runs, "alpha-new", "alpha", completed_at="2026-05-01T00:00:00+00:00")
    _write_run(runs, "alpha-newest-incomplete", "alpha", status="incomplete",
               completed_at="2026-06-01T00:00:00+00:00")
    _write_run(runs, "beta-run", "beta")

    summary = analyze_directory(runs, tmp_path / "analysis", selection="latest-complete")

    assert summary["run_ids"] == ["alpha-new", "beta-run"]
    assert any("excluded by selection" in warning for warning in summary["warnings"])


def test_explicit_run_ids_select_exactly_those_runs(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-old", "alpha")
    _write_run(runs, "alpha-new", "alpha")
    _write_run(runs, "beta-run", "beta")

    summary = analyze_directory(runs, tmp_path / "analysis", run_ids=["alpha-old"])

    assert summary["run_ids"] == ["alpha-old"]
    assert summary["models"] == ["alpha"]

    with pytest.raises(ValueError, match="not found"):
        analyze_directory(runs, tmp_path / "analysis", run_ids=["missing-run"])


def test_select_runs_prefers_complete_over_newer_incomplete(tmp_path):
    runs = tmp_path / "runs"
    complete = _write_run(runs, "alpha-done", "alpha", completed_at="2026-04-01T00:00:00+00:00")
    _write_run(runs, "alpha-bad", "alpha", status="incomplete", completed_at="2026-05-01T00:00:00+00:00")

    chosen = select_runs(discover_runs(runs), runs, selection="latest-complete")

    assert chosen == [complete]


def test_queue_and_analysis_outputs_are_not_treated_as_runs(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-run", "alpha")
    _write_run(runs, "queues/alpha-run/alpha", "alpha")
    _write_run(runs, "analysis/rerun/alpha-run", "alpha")

    assert [path.name for path in discover_runs(runs)] == ["alpha-run"]


def test_incomplete_runs_are_refused_without_allow_mixed(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-run", "alpha", status="incomplete")
    _write_run(runs, "beta-run", "beta")

    with pytest.raises(ValueError, match="incomplete runs cannot enter official comparison"):
        analyze_directory(runs, tmp_path / "analysis")


def test_figures_do_not_mix_incompatible_manifests(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-run", "alpha", parameter_size="7B")
    _write_run(runs, "beta-run", "beta", parameter_size="3B", profile_hash="other-profile")

    with pytest.raises(ValueError, match="incompatible evaluator/profile/dataset"):
        analyze_directory(runs, tmp_path / "analysis")


def test_figures_survive_a_single_renderer_failure(tmp_path, monkeypatch):
    from facets import figures

    runs = tmp_path / "runs"
    _write_run(runs, "alpha-run", "alpha", parameter_size="7B")
    _write_run(runs, "beta-run", "beta", parameter_size="3B")

    def _explode(**_kwargs):
        raise ValueError("render boom")

    survivors = tuple(
        _explode if renderer is figures.heatmap_model_task else renderer
        for renderer in figures.RENDERERS
    )
    monkeypatch.setattr(figures, "RENDERERS", survivors)

    summary = analyze_directory(runs, tmp_path / "analysis")

    assert "model_task_score_heatmap" not in summary["figures"]
    assert any("figure unavailable" in warning for warning in summary["warnings"])
    assert "record_score_distributions" in summary["figures"]


def test_analyze_cli_writes_index_and_rejects_conflicting_selection(tmp_path, capsys):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-run", "alpha", parameter_size="7B")
    _write_run(runs, "beta-run", "beta", parameter_size="3B")
    output = tmp_path / "analysis"

    assert main(["analyze", str(runs), "--output", str(output), "--latest-complete"]) == 0
    capsys.readouterr()

    assert (output / "index.md").exists()
    assert (output / "figures.json").exists()
    assert (output / "record_score_distributions.png").exists()

    conflict = main(["analyze", str(runs), "--output", str(output), "--run", "alpha-run",
                     "--latest-complete"])
    assert conflict == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_analyze_cli_returns_nonzero_on_ambiguous_cohort(tmp_path, capsys):
    runs = tmp_path / "runs"
    _write_run(runs, "alpha-old", "alpha")
    _write_run(runs, "alpha-new", "alpha")

    assert main(["analyze", str(runs), "--output", str(tmp_path / "analysis")]) == 2
    assert "ambiguous repeated models" in capsys.readouterr().err

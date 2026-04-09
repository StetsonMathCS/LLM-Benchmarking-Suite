"""
tui/app.py  —  LLM Benchmark Suite
Old 2-tab layout with all new features (weights, scoring, abort, live score, richer results).
"""
from __future__ import annotations

from dataclasses import asdict
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Optional
import urllib.request
import urllib.error

import yaml
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Rule,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from core.base import BenchmarkStatus, ModelConfig
from core.registry import ProviderRegistry
from core.suite import SuiteConfig, TestSuite
from core.scoring import (
    TASK_WEIGHTS as DEFAULT_TASK_WEIGHTS,
    PASS_THRESHOLD as DEFAULT_PASS_THRESHOLD,
    GRADE_THRESHOLDS,
)
from datasets.mapper import DatasetMapper


# ── Constants ────────────────────────────────────────────────────────────────

ORDERED_TASKS: list[str] = [
    "bug_fixing",
    "code_generation",
    "code_review",
    "partial_transform",
    "refactoring",
    "test_generation",
    "translation",
]

TASK_DISPLAY: dict[str, str] = {
    "bug_fixing":        "Bug Fixing",
    "code_generation":   "Code Generation",
    "code_review":       "Code Review",
    "partial_transform": "Partial Transform",
    "refactoring":       "Refactoring",
    "test_generation":   "Test Generation",
    "translation":       "Translation",
}

STATUS_ICON = {
    BenchmarkStatus.PASSED.value:  "[green]✓[/green]",
    BenchmarkStatus.FAILED.value:  "[red]✗[/red]",
    BenchmarkStatus.ERROR.value:   "[yellow]![/yellow]",
    BenchmarkStatus.RUNNING.value: "[cyan]…[/cyan]",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    return str(value)


def _parse_float(raw: str, fallback: float) -> float:
    try:
        return float(raw)
    except (ValueError, TypeError):
        return fallback


def _parse_int(raw: str, fallback: int) -> int:
    try:
        return int(raw)
    except (ValueError, TypeError):
        return fallback


# ── App ───────────────────────────────────────────────────────────────────────

class BenchmarkTUI(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "Python Benchmark Lab"
    SUB_TITLE = "Tune experiments · stream progress · save results"

    BINDINGS = [
        ("f1", "switch_tab('run-lab')",  "Run Lab"),
        ("f2", "switch_tab('results')",  "Results"),
        ("r",       "start_run",         "Start Run"),
        ("escape",  "abort_run",         "Abort"),
        ("ctrl+s",  "save_config",       "Save Config"),
        ("ctrl+r",  "refresh_results",   "Refresh"),
        ("ctrl+l",  "clear_log",         "Clear Log"),
        ("q",       "quit",              "Quit"),
    ]

    # ── Init ─────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__()
        self.root_dir    = Path(__file__).resolve().parent.parent
        self.config_file = self.root_dir / "config" / "model_config.yaml"
        self.mapper      = DatasetMapper(self.root_dir / "datasets")

        self._report_paths:    list[Path]            = []
        self._provider_values: list[str]             = []
        self._event_queue:     Queue[dict[str, Any]] = Queue()
        self._run_thread:      Optional[Thread]      = None
        self._running    = False
        self._abort_flag = False

        self._run_total     = 0
        self._run_completed = 0
        self._run_passed    = 0
        self._run_failed    = 0
        self._live_scores:  dict[str, list[float]] = {}

    # ── Compose ───────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with TabbedContent(id="main-tabs", initial="run-lab"):

            # ═══════════════════════════════════════════════
            # Tab 1 — Run Lab
            # ═══════════════════════════════════════════════
            with TabPane("Run Lab", id="run-lab"):
                with Horizontal(classes="pane-root"):

                    # ── Left: all controls ──────────────────
                    with VerticalScroll(id="controls", classes="panel"):
                        yield Static("Experiment Controls", classes="panel-title")

                        with Horizontal(classes="form-row"):
                            yield Label("Run Name",     classes="form-label")
                            yield Input(value="Python benchmark run", id="run-name", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Provider",     classes="form-label")
                            yield Select([], id="provider", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Model",        classes="form-label")
                            yield Input(placeholder="e.g. gpt-4o / llama3", id="model-name", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Base URL",     classes="form-label")
                            yield Input(placeholder="Optional", id="base-url", classes="form-field")
                            yield Button("⟳", id="fetch-ollama", classes="btn-small")
                        with Horizontal(classes="form-row", id="ollama-status-row"):
                            yield Label("",             classes="form-label")
                            yield Label("",             id="ollama-status", classes="form-field")
                        with Horizontal(classes="form-row", id="ollama-model-row"):
                            yield Label("Ollama Model", classes="form-label")
                            yield Select([], id="ollama-model-select", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("API Key",      classes="form-label")
                            yield Input(password=True, placeholder="Optional", id="api-key", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Temperature",  classes="form-label")
                            yield Input(value="0.7",    id="temperature", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Max Tokens",   classes="form-label")
                            yield Input(value="1024",   id="max-tokens", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("System Prompt",classes="form-label")
                            yield Input(placeholder="Optional", id="system-prompt", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Dataset Limit",classes="form-label")
                            yield Input(placeholder="blank = all rows", id="dataset-limit", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Translate To", classes="form-label")
                            yield Input(value="javascript", id="target-language", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Output Dir",   classes="form-label")
                            yield Input(value="reports/outputs/python", id="output-dir", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Dataset Root", classes="form-label")
                            yield Input(id="dataset-root", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Config File",  classes="form-label")
                            yield Input(id="config-file-path", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Pass Threshold", classes="form-label")
                            yield Input(value=f"{DEFAULT_PASS_THRESHOLD:.2f}", id="pass-threshold", classes="form-field")

                        yield Rule()
                        yield Static("Benchmarks  ·  select and weight", classes="section-title")

                        # Column header
                        with Horizontal(classes="wt-header"):
                            yield Static("Task",    classes="wt-col-task")
                            yield Static("On",      classes="wt-col-on")
                            yield Static("Weight",  classes="wt-col-weight")
                            yield Static("%",       classes="wt-col-pct")

                        with Container(id="benchmark-picks"):
                            for task in ORDERED_TASKS:
                                pct = f"{DEFAULT_TASK_WEIGHTS.get(task, 0)*100:.0f}%"
                                with Horizontal(classes="wt-row", id=f"row-{task}"):
                                    yield Label(TASK_DISPLAY.get(task, task), classes="wt-col-task")
                                    yield Checkbox("", id=f"bench-{task}", value=True, classes="wt-col-on")
                                    yield Input(
                                        value=f"{DEFAULT_TASK_WEIGHTS.get(task, 0.10):.3f}",
                                        id=f"weight-{task}",
                                        classes="wt-col-weight",
                                    )
                                    yield Static(pct, id=f"pct-{task}", classes="wt-col-pct")

                        with Horizontal(classes="wt-footer"):
                            yield Static("Sum:", classes="wt-sum-label")
                            yield Static("1.000  ✓", id="weight-sum", classes="wt-sum-ok")

                        with Horizontal(classes="weight-btns"):
                            yield Button("✓ All",       id="select-all",        classes="btn-small")
                            yield Button("✗ None",      id="deselect-all",      classes="btn-small")
                            yield Button("↺ Reset",     id="reset-weights",     classes="btn-small")
                            yield Button("⊜ Normalize", id="normalize-weights", classes="btn-small")

                        yield Rule()
                        with Horizontal(id="run-actions"):
                            yield Button("Load Config",  id="load-config")
                            yield Button("Save Config",  id="save-config",  variant="success")
                            yield Button("Start Run",    id="start-run",    variant="primary")
                            yield Button("■ Abort",      id="abort-run",    classes="btn-danger")

                    # ── Right: live feed ────────────────────
                    with VerticalScroll(id="live-pane", classes="panel"):
                        yield Static("Live Run Feed", classes="panel-title")
                        yield Static("●  IDLE", id="run-status", classes="status-banner status-idle")

                        with Horizontal(classes="card-row"):
                            yield Static("Total\n—",   id="card-total",     classes="card")
                            yield Static("Done\n—",    id="card-completed", classes="card")
                            yield Static("Passed\n—",  id="card-passed",    classes="card card-pass")
                            yield Static("Failed\n—",  id="card-failed",    classes="card card-fail")
                            yield Static("Score\n—",   id="card-score",     classes="card card-score")
                            yield Static("Grade\n—",   id="card-grade",     classes="card card-grade")

                        yield ProgressBar(total=100, show_eta=False, id="run-progress")

                        yield DataTable(id="run-table")
                        yield RichLog(id="run-log", highlight=True, wrap=True, markup=True)

            # ═══════════════════════════════════════════════
            # Tab 2 — Saved Results
            # ═══════════════════════════════════════════════
            with TabPane("Saved Results", id="results"):
                with VerticalScroll(classes="pane-root"):
                    with Horizontal(classes="form-row"):
                        yield Label("Results Dir",  classes="form-label")
                        yield Input(value="reports/outputs/python", id="results-dir", classes="form-field")
                        yield Button("↺ Refresh", id="refresh-results")
                    yield DataTable(id="results-table")
                    yield RichLog(id="results-preview", highlight=True, wrap=True, markup=True)

        yield Footer()

    # ── Mount ─────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        print(f"DEBUG on_mount: _running={self._running}, _run_thread={self._run_thread}")
        self._init_tables()
        self._load_provider_options()
        self._populate_path_fields()
        self._load_model_config_into_form()
        self._refresh_results()
        self.set_interval(0.2, self._drain_events)
        # Hide Ollama-specific rows until a successful connection
        self.query_one("#ollama-status-row").display = False
        self.query_one("#ollama-model-row").display  = False
        print(f"DEBUG on_mount done: _running={self._running}, _run_thread={self._run_thread}")

    def _init_tables(self) -> None:
        t = self.query_one("#run-table", DataTable)
        t.cursor_type = "row"
        t.add_columns("Task", "Record", "Status", "Score", "Time")

        r = self.query_one("#results-table", DataTable)
        r.cursor_type = "row"
        r.add_columns("Run Name", "Score", "Grade", "Model", "Date", "KB")

    def _populate_path_fields(self) -> None:
        self.query_one("#dataset-root",    Input).value = str(self.root_dir / "datasets")
        self.query_one("#config-file-path",Input).value = str(self.config_file)

    # ── Providers ─────────────────────────────────────────────────────────

    def _load_provider_options(self) -> None:
        providers: list[str] = []
        providers_dir = self.root_dir / "providers"
        if providers_dir.exists():
            for child in sorted(providers_dir.iterdir()):
                if child.is_dir() and (child / "provider.py").exists():
                    providers.append(child.name)
        if not providers:
            providers = ["ollama"]
        self._provider_values = providers
        self.query_one("#provider", Select).set_options([(p, p) for p in providers])

    # ── Ollama model discovery ────────────────────────────────────────────

    def _fetch_ollama_models(self) -> None:
        base_url = self.query_one("#base-url", Input).value.strip().rstrip("/")
        if not base_url:
            self.notify("Enter a Base URL first.", severity="warning")
            return

        self.query_one("#ollama-status-row").display = True
        self.query_one("#ollama-model-row").display  = False
        status = self.query_one("#ollama-status", Label)
        status.update("⟳ Connecting …")
        status.remove_class("ollama-ok", "ollama-err")

        def worker() -> None:
            try:
                url = f"{base_url}/api/tags"
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                self._event_queue.put({"type": "ollama_models", "models": models})
            except Exception as exc:
                self._event_queue.put({"type": "ollama_models", "error": str(exc)})

        Thread(target=worker, daemon=True).start()

    # ── Config load / save ────────────────────────────────────────────────

    def _config_file_path(self) -> Path:
        raw = self.query_one("#config-file-path", Input).value.strip()
        return Path(raw) if raw else self.config_file

    def _load_config_file(self) -> dict:
        path = self._config_file_path()
        if not path.exists():
            return {}
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _load_model_config_into_form(self) -> None:
        cfg = self._load_config_file()
        provider = str(cfg.get("provider", "ollama"))
        if provider not in self._provider_values and self._provider_values:
            provider = self._provider_values[0]
        self.query_one("#provider",      Select).value = provider
        self.query_one("#model-name",    Input).value  = str(cfg.get("model_name",    "") or "")
        self.query_one("#base-url",      Input).value  = str(cfg.get("base_url",      "") or "")
        self.query_one("#api-key",       Input).value  = str(cfg.get("api_key",       "") or "")
        self.query_one("#temperature",   Input).value  = str(cfg.get("temperature",   0.7))
        self.query_one("#max-tokens",    Input).value  = str(cfg.get("max_tokens",    1024))
        self.query_one("#system-prompt", Input).value  = str(cfg.get("system_prompt", "") or "")

    def _model_config_from_form(self) -> ModelConfig:
        return ModelConfig(
            provider      = str(self.query_one("#provider",      Select).value),
            model_name    = self.query_one("#model-name",    Input).value.strip(),
            base_url      = self.query_one("#base-url",      Input).value.strip() or None,
            api_key       = self.query_one("#api-key",       Input).value.strip() or None,
            temperature   = _parse_float(self.query_one("#temperature",   Input).value.strip(), 0.7),
            max_tokens    = _parse_int(  self.query_one("#max-tokens",    Input).value.strip(), 1024),
            system_prompt = self.query_one("#system-prompt", Input).value.strip() or None,
            extra_params  = {},
        )

    def _save_model_config(self) -> None:
        cfg  = self._model_config_from_form()
        path = self._config_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
        self.notify(f"Saved → {path}", title="Config Saved")

    # ── Benchmark / weight helpers ────────────────────────────────────────

    def _available_tasks(self) -> list[str]:
        return [
            t for t in ORDERED_TASKS
            if "python" in self.mapper.TASK_DATASET_MAP.get(t, {}).get("languages", [])
        ]

    def _selected_tasks(self) -> list[str]:
        return [t for t in self._available_tasks() if self.query_one(f"#bench-{t}", Checkbox).value]

    def _task_weights_from_form(self) -> dict[str, float]:
        return {
            task: _parse_float(self.query_one(f"#weight-{task}", Input).value.strip(),
                               DEFAULT_TASK_WEIGHTS.get(task, 0.10))
            for task in ORDERED_TASKS
        }

    def _refresh_weight_sum_label(self) -> None:
        weights = self._task_weights_from_form()
        total   = sum(weights.values())
        ok      = abs(total - 1.0) < 0.001
        label   = self.query_one("#weight-sum", Static)
        if ok:
            label.update(f"{total:.3f}  ✓")
            label.remove_class("wt-sum-bad")
            label.add_class("wt-sum-ok")
        else:
            label.update(f"{total:.3f}  ✗  (must equal 1.000)")
            label.remove_class("wt-sum-ok")
            label.add_class("wt-sum-bad")
        for task in ORDERED_TASKS:
            w   = weights.get(task, 0.0)
            pct = f"{w*100:.1f}%" if total > 0 else "—"
            self.query_one(f"#pct-{task}", Static).update(pct)

    # ── Run helpers ───────────────────────────────────────────────────────

    def _dataset_limit(self) -> Optional[int]:
        raw = self.query_one("#dataset-limit", Input).value.strip()
        v   = _parse_int(raw, 0)
        return v if v > 0 else None

    def _pass_threshold(self) -> float:
        raw = self.query_one("#pass-threshold", Input).value.strip()
        return max(0.0, min(1.0, _parse_float(raw, DEFAULT_PASS_THRESHOLD)))

    def _safe_output_dir(self) -> Path:
        raw  = self.query_one("#output-dir", Input).value.strip() or "reports/outputs/python"
        path = Path(raw) if Path(raw).is_absolute() else (self.root_dir / raw).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _dataset_root(self) -> Path:
        raw = self.query_one("#dataset-root", Input).value.strip()
        return Path(raw) if raw else (self.root_dir / "datasets")

    def _estimate_total(self, selected: list[str], limit: Optional[int]) -> int:
        total = 0
        for task in selected:
            try:
                info  = self.mapper.get_dataset_info(task, "python")
                count = int(info["record_count"])
                total += min(count, limit) if limit else count
            except Exception:
                continue
        return total

    # ── Stat cards ────────────────────────────────────────────────────────

    def _set_card(self, card_id: str, title: str, value: str) -> None:
        self.query_one(card_id, Static).update(f"{title}\n{value}")

    def _reset_run_state(self) -> None:
        self._run_completed = 0
        self._run_passed    = 0
        self._run_failed    = 0
        self._live_scores   = {}
        self._set_card("#card-completed", "Done",   "0")
        self._set_card("#card-passed",    "Passed", "0")
        self._set_card("#card-failed",    "Failed", "0")
        self._set_card("#card-score",     "Score",  "—")
        self._set_card("#card-grade",     "Grade",  "—")
        self.query_one("#run-progress", ProgressBar).progress = 0

    def _update_live_score(self) -> None:
        if not self._live_scores:
            return
        weights = self._task_weights_from_form()
        total_w = sum(weights.get(t, 0.0) for t in self._live_scores if self._live_scores[t])
        if total_w == 0:
            return
        wsum  = sum(weights.get(t, 0.0) * (sum(v) / len(v)) for t, v in self._live_scores.items() if v)
        score = wsum / total_w
        grade = next(g for thr, g in GRADE_THRESHOLDS if score >= thr)
        self._set_card("#card-score", "Score", f"{score:.1%}")
        self._set_card("#card-grade", "Grade", grade)

    # ── Progress callback (worker thread) ─────────────────────────────────

    def _on_progress_event(
        self,
        task_name: str,
        record_id: str,
        status:    BenchmarkStatus,
        score:     Optional[float],
    ) -> None:
        self._event_queue.put({
            "type":   "progress",
            "task":   task_name,
            "record": str(record_id),
            "status": status.value if isinstance(status, BenchmarkStatus) else str(status),
            "score":  score,
            "at":     datetime.now().strftime("%H:%M:%S"),
        })

    # ── Worker ────────────────────────────────────────────────────────────

    def _run_suite_worker(
        self,
        model_cfg:       ModelConfig,
        selected:        list[str],
        limit:           Optional[int],
        run_name:        str,
        target_language: str,
        output_dir:      Path,
        dataset_root:    Path,
        task_weights:    dict[str, float],
        pass_threshold:  float,
    ) -> None:
        try:
            provider  = ProviderRegistry.create(model_cfg)
            mapper    = DatasetMapper(dataset_root)
            suite_cfg = SuiteConfig(
                name                = run_name,
                selected_benchmarks = selected,
                language            = "python",
                output_dir          = str(output_dir),
                progress_callback   = self._on_progress_event,
                target_language     = target_language or None,
                task_weights        = task_weights,
                pass_threshold      = pass_threshold,
            )
            suite = TestSuite(provider, suite_cfg)
            suite.register_benchmarks()
            suite._mapper = mapper

            if limit:
                orig = suite._mapper.load_dataset
                suite._mapper.load_dataset = lambda t, l, **kw: orig(t, l, limit=limit, **kw)  # type: ignore

            results  = suite.run_all()
            summary  = suite.get_summary()
            save_path = self._save_run_results(
                output_dir, run_name, model_cfg, selected, limit, summary, results
            )
            self._event_queue.put({"type": "done", "summary": summary, "save_path": str(save_path)})

        except BaseException as exc:
            self._event_queue.put({"type": "error", "message": str(exc)})

    # ── Save results ──────────────────────────────────────────────────────

    def _save_run_results(
        self,
        output_dir: Path,
        run_name:   str,
        model_cfg:  ModelConfig,
        selected:   list[str],
        limit:      Optional[int],
        summary:    dict[str, Any],
        results:    list[Any],
    ) -> Path:
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug     = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name).strip("_") or "run"
        out_path = output_dir / f"{stamp}_{slug}.json"
        payload  = {
            "run_name":            run_name,
            "timestamp":           datetime.now().isoformat(timespec="seconds"),
            "language":            "python",
            "selected_benchmarks": selected,
            "dataset_limit":       limit,
            "model_config":        to_jsonable(model_cfg),
            "summary":             to_jsonable(summary),
            "results":             [to_jsonable(r) for r in results],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    # ── Event drain ───────────────────────────────────────────────────────

    def _drain_events(self) -> None:  # noqa: C901
        # Auto-recover if the worker thread died without sending a done/error event
        if (
            self._running
            and self._run_thread is not None
            and not self._run_thread.is_alive()
            and self._event_queue.empty()
        ):
            self._running = False
            self.query_one("#start-run", Button).disabled = False
            status_w = self.query_one("#run-status", Static)
            status_w.update("✗  ERROR  ·  Worker thread exited unexpectedly")
            status_w.remove_class("status-idle", "status-running", "status-done")
            status_w.add_class("status-error")
            self.notify("Run ended unexpectedly. Ready to start again.", severity="warning")

        while True:
            try:
                event = self._event_queue.get_nowait()
            except Empty:
                break

            etype = event.get("type")

            if etype == "progress":
                if self._abort_flag:
                    continue

                self._run_completed += 1
                status_val = str(event["status"])
                score      = event.get("score")

                if status_val == BenchmarkStatus.PASSED.value:
                    self._run_passed += 1
                elif status_val in (BenchmarkStatus.FAILED.value, BenchmarkStatus.ERROR.value):
                    self._run_failed += 1

                if score is not None:
                    self._live_scores.setdefault(event["task"], []).append(score)
                    self._update_live_score()

                self._set_card("#card-completed", "Done",   str(self._run_completed))
                self._set_card("#card-passed",    "Passed", str(self._run_passed))
                self._set_card("#card-failed",    "Failed", str(self._run_failed))

                if self._run_total > 0:
                    pb = self.query_one("#run-progress", ProgressBar)
                    pb.total    = self._run_total
                    pb.progress = self._run_completed

                icon      = STATUS_ICON.get(status_val, status_val)
                score_str = f"{score:.2f}" if score is not None else "—"
                self.query_one("#run-table", DataTable).add_row(
                    event["task"], event["record"], icon, score_str, event["at"],
                )
                log_color = "green" if status_val == "passed" else ("red" if status_val in ("failed", "error") else "yellow")
                self.query_one("#run-log", RichLog).write(
                    f"[dim]{event['at']}[/dim]  "
                    f"[cyan]{event['task']}[/cyan] [dim]#{event['record']}[/dim]  "
                    f"[{log_color}]{status_val}[/{log_color}]"
                    + (f"  [dim]{score_str}[/dim]" if score is not None else "")
                )

            elif etype == "done":
                self._running = False
                self.query_one("#start-run", Button).disabled = False
                summary     = event["summary"]
                final_score = summary.get("final_score", 0.0)
                grade       = summary.get("grade", "?")
                elapsed     = summary.get("elapsed_s", 0.0)

                status_w = self.query_one("#run-status", Static)
                status_w.update(f"✓  DONE  ·  Score: {final_score:.1%}  ·  Grade: {grade}  ·  {elapsed:.1f}s")
                status_w.remove_class("status-idle", "status-running", "status-error")
                status_w.add_class("status-done")

                self._set_card("#card-score", "Score", f"{final_score:.1%}")
                self._set_card("#card-grade", "Grade", grade)

                log = self.query_one("#run-log", RichLog)
                log.write("")
                log.write("[bold cyan]" + "─" * 44 + "[/bold cyan]")
                log.write("[bold]  Task Scores[/bold]")
                log.write("[bold cyan]" + "─" * 44 + "[/bold cyan]")
                for task in ORDERED_TASKS:
                    ts = summary.get("task_scores", {}).get(task)
                    if not ts:
                        continue
                    fill = round(ts["mean_score"] * 20)
                    bar  = "[green]" + "█" * fill + "[/green][dim]" + "░" * (20 - fill) + "[/dim]"
                    log.write(
                        f"  [cyan]{TASK_DISPLAY.get(task, task):<20}[/cyan] "
                        f"{bar}  [bold]{ts['mean_score']:.3f}[/bold]"
                        f"  [dim]n={ts['scored_count']} pass={ts['pass_rate']:.0%}[/dim]"
                    )
                cat_scores = summary.get("category_scores", {})
                if cat_scores:
                    log.write("[bold cyan]" + "─" * 44 + "[/bold cyan]")
                    for cat, sc in cat_scores.items():
                        log.write(f"  [magenta]{cat.capitalize():<16}[/magenta]  {sc:.3f}")
                log.write("[bold cyan]" + "─" * 44 + "[/bold cyan]")
                log.write(f"  [bold white]Final Score  {final_score:.1%}   Grade  {grade}[/bold white]")
                log.write("[bold cyan]" + "─" * 44 + "[/bold cyan]")
                log.write(f"[dim]Saved → {event['save_path']}[/dim]")

                self.notify(f"Score {final_score:.1%}  ·  Grade {grade}", title="Run Complete")
                self.query_one("#results-dir", Input).value = self.query_one("#output-dir", Input).value
                self._refresh_results()

            elif etype == "error":
                self._running = False
                self.query_one("#start-run", Button).disabled = False
                status_w = self.query_one("#run-status", Static)
                status_w.update(f"✗  ERROR  ·  {event['message'][:80]}")
                status_w.remove_class("status-idle", "status-running", "status-done")
                status_w.add_class("status-error")
                self.query_one("#run-log", RichLog).write(f"[bold red]ERROR:[/bold red] {event['message']}")
                self.notify(event["message"], title="Run Error", severity="error")

            elif etype == "ollama_models":
                status_lbl  = self.query_one("#ollama-status", Label)
                status_row  = self.query_one("#ollama-status-row")
                model_row   = self.query_one("#ollama-model-row")
                status_row.display = True
                if "error" in event:
                    status_lbl.update(f"✗  {event['error'][:80]}")
                    status_lbl.remove_class("ollama-ok")
                    status_lbl.add_class("ollama-err")
                    model_row.display = False
                else:
                    models = event["models"]
                    status_lbl.update(f"✓  Connected — {len(models)} model(s) available")
                    status_lbl.remove_class("ollama-err")
                    status_lbl.add_class("ollama-ok")
                    if models:
                        sel = self.query_one("#ollama-model-select", Select)
                        sel.set_options([(m, m) for m in models])
                        model_row.display = True
                    else:
                        model_row.display = False

    # ── Start / Abort ─────────────────────────────────────────────────────

    def _start_run(self) -> None:
        if self._running:
            # If the thread already died without sending a done/error event, recover silently.
            if self._run_thread is not None and not self._run_thread.is_alive():
                self._running = False
                self.query_one("#start-run", Button).disabled = False
            else:
                self.notify("A run is already in progress.", severity="warning")
                return

        model_cfg = self._model_config_from_form()
        if not model_cfg.model_name:
            self.notify("Model Name is required.", severity="error")
            return

        selected = self._selected_tasks()
        if not selected:
            self.notify("Select at least one benchmark.", severity="error")
            return

        weights = self._task_weights_from_form()
        if abs(sum(weights.values()) - 1.0) > 0.01:
            self.notify("Task weights must sum to 1.000. Use ⊜ Normalize to fix.", severity="warning")
            return

        limit           = self._dataset_limit()
        pass_threshold  = self._pass_threshold()
        run_name        = self.query_one("#run-name",        Input).value.strip() or "Python benchmark run"
        target_language = self.query_one("#target-language", Input).value.strip()
        output_dir      = self._safe_output_dir()
        dataset_root    = self._dataset_root()

        self.query_one("#run-table", DataTable).clear()
        self.query_one("#run-log",   RichLog).clear()
        self._run_total = self._estimate_total(selected, limit)
        self._set_card("#card-total", "Total", str(self._run_total))
        self._reset_run_state()

        status_w = self.query_one("#run-status", Static)
        status_w.update("◌  RUNNING …")
        status_w.remove_class("status-idle", "status-done", "status-error")
        status_w.add_class("status-running")

        pb = self.query_one("#run-progress", ProgressBar)
        pb.total    = max(self._run_total, 1)
        pb.progress = 0

        self.query_one("#start-run", Button).disabled = True
        self._abort_flag = False
        self._running    = True

        log = self.query_one("#run-log", RichLog)
        log.write(f"[bold cyan]Run:[/bold cyan] {run_name}")
        log.write(f"[dim]Model:  {model_cfg.model_name}  ({model_cfg.provider})[/dim]")
        log.write(f"[dim]Tasks:  {', '.join(selected)}[/dim]")
        log.write(f"[dim]Limit:  {limit or 'all rows'}   Threshold: {pass_threshold:.2f}[/dim]")
        log.write(f"[dim]Output: {output_dir}[/dim]")
        log.write("")

        self._run_thread = Thread(
            target=self._run_suite_worker,
            kwargs=dict(
                model_cfg       = model_cfg,
                selected        = selected,
                limit           = limit,
                run_name        = run_name,
                target_language = target_language,
                output_dir      = output_dir,
                dataset_root    = dataset_root,
                task_weights    = weights,
                pass_threshold  = pass_threshold,
            ),
            daemon=True,
        )
        try:
            self._run_thread.start()
        except Exception as exc:
            self._running = False
            self.query_one("#start-run", Button).disabled = False
            status_w = self.query_one("#run-status", Static)
            status_w.update(f"✗  ERROR  ·  {str(exc)[:80]}")
            status_w.remove_class("status-idle", "status-running", "status-done")
            status_w.add_class("status-error")
            self.notify(str(exc), title="Failed to Start Run", severity="error")

    def _abort_run(self) -> None:
        if not self._running:
            return
        self._abort_flag = True
        self._running    = False
        self.query_one("#start-run", Button).disabled = False
        status_w = self.query_one("#run-status", Static)
        status_w.update("⊘  ABORTED")
        status_w.remove_class("status-running")
        status_w.add_class("status-error")
        self.query_one("#run-log", RichLog).write("[yellow]Run aborted by user.[/yellow]")
        self.notify("Run aborted.", severity="warning")

    # ── Results tab ───────────────────────────────────────────────────────

    def _refresh_results(self) -> None:
        raw  = self.query_one("#results-dir", Input).value.strip() or "reports/outputs/python"
        base = Path(raw) if Path(raw).is_absolute() else (self.root_dir / raw).resolve()

        self._report_paths = []
        if base.exists():
            self._report_paths = sorted(base.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        table = self.query_one("#results-table", DataTable)
        table.clear()
        for idx, path in enumerate(self._report_paths):
            meta  = self._load_result_meta(path)
            stat  = path.stat()
            score = f"{meta['final_score']:.1%}" if meta.get("final_score") is not None else "—"
            table.add_row(
                meta.get("run_name", path.stem)[:40],
                score,
                meta.get("grade", "—"),
                meta.get("model", "—")[:20],
                datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                f"{stat.st_size / 1024:.1f}",
                key=str(idx),
            )

        preview = self.query_one("#results-preview", RichLog)
        preview.clear()
        if not self._report_paths:
            preview.write("[dim]No saved run files found in this folder.[/dim]")

    def _load_result_meta(self, path: Path) -> dict:
        try:
            data    = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            summary = data.get("summary", {})
            return {
                "run_name":    data.get("run_name", path.stem),
                "final_score": summary.get("final_score"),
                "grade":       summary.get("grade"),
                "model":       data.get("model_config", {}).get("model_name", ""),
            }
        except Exception:
            return {}

    def _preview_saved_file(self, path: Path) -> None:
        preview = self.query_one("#results-preview", RichLog)
        preview.clear()
        try:
            data    = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            summary = data.get("summary", {})
            final   = summary.get("final_score")
            grade   = summary.get("grade", "?")
            elapsed = summary.get("elapsed_s", 0)

            preview.write(f"[bold cyan]{data.get('run_name', '')}[/bold cyan]  [dim]{data.get('timestamp', '')}[/dim]")
            preview.write(
                f"[bold]Score:[/bold] {f'{final:.1%}' if final is not None else '—'}"
                f"  [bold]Grade:[/bold] {grade}"
                f"  [bold]Elapsed:[/bold] {elapsed:.1f}s"
            )
            mcfg = data.get("model_config", {})
            preview.write(f"[bold]Model:[/bold] {mcfg.get('model_name', '—')}  [bold]Provider:[/bold] {mcfg.get('provider', '—')}")
            preview.write("")

            task_scores = summary.get("task_scores", {})
            if task_scores:
                preview.write("[bold]Task Scores[/bold]")
                for task in ORDERED_TASKS:
                    ts = task_scores.get(task)
                    if not ts:
                        continue
                    fill = round(ts["mean_score"] * 20)
                    bar  = "█" * fill + "░" * (20 - fill)
                    preview.write(
                        f"  [cyan]{TASK_DISPLAY.get(task, task):<20}[/cyan] "
                        f"[green]{bar}[/green]  {ts['mean_score']:.3f}"
                        f"  [dim]n={ts['scored_count']} pass={ts['pass_rate']:.0%}[/dim]"
                    )

            cat_scores = summary.get("category_scores", {})
            if cat_scores:
                preview.write("")
                preview.write("[bold]Category Scores[/bold]")
                for cat, sc in cat_scores.items():
                    preview.write(f"  [magenta]{cat.capitalize():<16}[/magenta] {sc:.3f}")

            preview.write("")
            preview.write("[dim]── Raw JSON ──────────────────────────[/dim]")
            raw_text = json.dumps(data, indent=2)
            for line in raw_text.splitlines()[:120]:
                preview.write(line)
            if len(raw_text.splitlines()) > 120:
                preview.write("[dim]… truncated …[/dim]")

        except Exception as e:
            preview.write(f"[red]Could not parse file: {e}[/red]")

    # ── Event handlers ────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id and event.input.id.startswith("weight-"):
            self._refresh_weight_sum_label()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "ollama-model-select" and event.value is not Select.BLANK:
            self.query_one("#model-name", Input).value = str(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:  # noqa: C901
        bid = event.button.id
        if   bid == "start-run":
            print(f"DEBUG: Start Run button clicked. _running={self._running}, thread={self._run_thread}, disabled={event.button.disabled}")
            self._start_run()
        elif bid == "abort-run":       self._abort_run()
        elif bid == "fetch-ollama":    self._fetch_ollama_models()
        elif bid == "load-config":     self._load_model_config_into_form(); self.notify("Config loaded.")
        elif bid == "save-config":     self._save_model_config()
        elif bid == "refresh-results": self._refresh_results(); self.notify("Results refreshed.")
        elif bid == "clear-log":       self.query_one("#run-log", RichLog).clear()
        elif bid == "select-all":
            for t in self._available_tasks():
                self.query_one(f"#bench-{t}", Checkbox).value = True
        elif bid == "deselect-all":
            for t in self._available_tasks():
                self.query_one(f"#bench-{t}", Checkbox).value = False
        elif bid == "reset-weights":
            for t in ORDERED_TASKS:
                self.query_one(f"#weight-{t}", Input).value = f"{DEFAULT_TASK_WEIGHTS.get(t, 0.10):.3f}"
            self._refresh_weight_sum_label()
            self.notify("Weights reset to defaults.")
        elif bid == "normalize-weights":
            weights = self._task_weights_from_form()
            total   = sum(weights.values())
            if total > 0:
                for t, w in weights.items():
                    self.query_one(f"#weight-{t}", Input).value = f"{w / total:.3f}"
            self._refresh_weight_sum_label()
            self.notify("Weights normalized to 1.000.")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "results-table":
            return
        key = event.row_key.value if event.row_key else None
        if key is None:
            return
        idx = int(key)
        if 0 <= idx < len(self._report_paths):
            self._preview_saved_file(self._report_paths[idx])

    # ── Actions ───────────────────────────────────────────────────────────

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one("#main-tabs", TabbedContent).active = tab_id

    def action_start_run(self)       -> None: self._start_run()
    def action_abort_run(self)       -> None: self._abort_run()
    def action_save_config(self)     -> None: self._save_model_config()
    def action_refresh_results(self) -> None: self._refresh_results()
    def action_clear_log(self)       -> None: self.query_one("#run-log", RichLog).clear()


if __name__ == "__main__":
    BenchmarkTUI().run()

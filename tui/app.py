from __future__ import annotations

from dataclasses import asdict
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

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
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)
from core.base import BenchmarkStatus, ModelConfig
from core.registry import ProviderRegistry
from core.suite import SuiteConfig, TestSuite
from datasets.mapper import DatasetMapper


def to_jsonable(value: Any) -> Any:
    """Convert nested values (dataclasses/enums/paths) into JSON-safe primitives."""
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


class BenchmarkTUI(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "Python Benchmark Lab"
    SUB_TITLE = "Tune experiments, stream progress, save wherever you want"

    BINDINGS = [
        ("r", "start_run", "Run"),
        ("ctrl+l", "clear_log", "Clear Log"),
        ("ctrl+s", "save_model_config", "Save Model"),
        ("ctrl+r", "refresh_results", "Refresh Results"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.root_dir = Path(__file__).resolve().parent.parent
        self.config_file = self.root_dir / "config" / "model_config.yaml"
        self.mapper = DatasetMapper(self.root_dir / "datasets")

        self._report_paths: list[Path] = []
        self._provider_values: list[str] = []
        self._event_queue: Queue[dict[str, Any]] = Queue()
        self._run_thread: Thread | None = None
        self._running = False

        self._run_total = 0
        self._run_completed = 0
        self._run_passed = 0
        self._run_failed = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="main-tabs", initial="run-lab"):
            with TabPane("Run Lab", id="run-lab"):
                with Horizontal(classes="pane-root"):
                    with VerticalScroll(id="controls", classes="panel"):
                        yield Static("Experiment Controls", classes="panel-title")
                        with Horizontal(classes="form-row"):
                            yield Label("Run Name", classes="form-label")
                            yield Input(value="Python benchmark run", id="run-name", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Provider", classes="form-label")
                            yield Select([], id="provider", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Model", classes="form-label")
                            yield Input(placeholder="e.g. gpt-5-nano", id="model-name", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Base URL", classes="form-label")
                            yield Input(placeholder="Optional", id="base-url", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("API Key", classes="form-label")
                            yield Input(password=True, placeholder="Optional", id="api-key", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Temperature", classes="form-label")
                            yield Input(value="0.7", id="temperature", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Max Tokens", classes="form-label")
                            yield Input(value="1024", id="max-tokens", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("System Prompt", classes="form-label")
                            yield Input(placeholder="Optional", id="system-prompt", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Dataset Limit", classes="form-label")
                            yield Input(value="", placeholder="Empty = all rows", id="dataset-limit", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Translate To", classes="form-label")
                            yield Input(value="javascript", id="target-language", classes="form-field")
                        with Horizontal(classes="form-row"):
                            yield Label("Save Dir", classes="form-label")
                            yield Input(value="reports/outputs/python", id="output-dir", classes="form-field")

                        yield Static("Benchmarks (Python)", classes="section-title")
                        with Container(id="benchmark-picks"):
                            # Checkboxes are populated in on_mount.
                            pass

                        with Horizontal(id="run-actions"):
                            yield Button("Load Model Config", id="load-config")
                            yield Button("Save Model Config", id="save-config", variant="success")
                            yield Button("Start Run", id="start-run", variant="primary")

                    with VerticalScroll(id="live-pane", classes="panel"):
                        yield Static("Live Run Feed", classes="panel-title")
                        yield Static("Idle", id="run-status", classes="status-banner")
                        with Horizontal(classes="card-row"):
                            yield Static("Total\n0", id="card-total", classes="card")
                            yield Static("Completed\n0", id="card-completed", classes="card")
                            yield Static("Passed\n0", id="card-passed", classes="card")
                            yield Static("Failed\n0", id="card-failed", classes="card")
                        yield DataTable(id="run-table")
                        yield RichLog(id="run-log", highlight=True, wrap=True)

            with TabPane("Saved Results", id="results"):
                with VerticalScroll(classes="pane-root"):
                    with Horizontal(classes="form-row"):
                        yield Label("Results Dir", classes="form-label")
                        yield Input(value="reports/outputs/python", id="results-dir", classes="form-field")
                        yield Button("Refresh", id="refresh-results")
                    yield DataTable(id="results-table")
                    yield Static("Select a row to preview saved run data.", id="results-preview")

        yield Footer()

    def on_mount(self) -> None:
        self._init_tables()
        self._load_provider_options()
        self._load_benchmark_checks()
        self._load_model_config_into_form()
        self._refresh_results()
        self.set_interval(0.2, self._drain_events)

    def _init_tables(self) -> None:
        run_table = self.query_one("#run-table", DataTable)
        run_table.cursor_type = "row"
        run_table.add_columns("Task", "Record", "Status", "When")

        results_table = self.query_one("#results-table", DataTable)
        results_table.cursor_type = "row"
        results_table.add_columns("File", "Modified", "Size (KB)")

    def _load_provider_options(self) -> None:
        providers: list[str] = []
        providers_dir = self.root_dir / "providers"
        if providers_dir.exists():
            for child in providers_dir.iterdir():
                if child.is_dir() and (child / "provider.py").exists():
                    providers.append(child.name)
        providers = sorted(providers)
        if not providers:
            providers = ["ollama"]
        self._provider_values = providers
        self.query_one("#provider", Select).set_options([(p, p) for p in providers])

    def _available_python_benchmarks(self) -> list[str]:
        ordered = [
            "bug_fixing",
            "code_completion",
            "code_generation",
            "code_review",
            "partial_transform",
            "refactoring",
            "test_generation",
            "translation",
        ]
        valid = []
        for task in ordered:
            task_cfg = self.mapper.TASK_DATASET_MAP.get(task)
            if not task_cfg:
                continue
            if "python" in task_cfg.get("languages", []):
                valid.append(task)
        return valid

    def _load_benchmark_checks(self) -> None:
        picks = self.query_one("#benchmark-picks", Container)
        for task in self._available_python_benchmarks():
            picks.mount(Checkbox(task, id=f"bench-{task}", value=True))

    def _parse_float(self, raw: str, fallback: float) -> float:
        try:
            return float(raw)
        except ValueError:
            return fallback

    def _parse_int(self, raw: str, fallback: int) -> int:
        try:
            return int(raw)
        except ValueError:
            return fallback

    def _selected_benchmarks(self) -> list[str]:
        selected: list[str] = []
        for task in self._available_python_benchmarks():
            cb = self.query_one(f"#bench-{task}", Checkbox)
            if cb.value:
                selected.append(task)
        return selected

    def _dataset_limit(self) -> int | None:
        raw = self.query_one("#dataset-limit", Input).value.strip()
        if not raw:
            return None
        limit = self._parse_int(raw, 0)
        return limit if limit > 0 else None

    def _model_config_from_form(self) -> ModelConfig:
        return ModelConfig(
            provider=str(self.query_one("#provider", Select).value),
            model_name=self.query_one("#model-name", Input).value.strip(),
            base_url=self.query_one("#base-url", Input).value.strip() or None,
            api_key=self.query_one("#api-key", Input).value.strip() or None,
            temperature=self._parse_float(self.query_one("#temperature", Input).value.strip(), 0.7),
            max_tokens=self._parse_int(self.query_one("#max-tokens", Input).value.strip(), 1024),
            system_prompt=self.query_one("#system-prompt", Input).value.strip() or None,
            extra_params={},
        )

    def _load_config_file(self) -> dict:
        if not self.config_file.exists():
            return {}
        try:
            return yaml.safe_load(self.config_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _load_model_config_into_form(self) -> None:
        cfg = self._load_config_file()
        provider_widget = self.query_one("#provider", Select)
        provider = str(cfg.get("provider", "ollama"))
        if provider not in self._provider_values and self._provider_values:
            provider = self._provider_values[0]
        provider_widget.value = provider

        self.query_one("#model-name", Input).value = str(cfg.get("model_name", ""))
        self.query_one("#base-url", Input).value = str(cfg.get("base_url", "") or "")
        self.query_one("#api-key", Input).value = str(cfg.get("api_key", "") or "")
        self.query_one("#temperature", Input).value = str(cfg.get("temperature", 0.7))
        self.query_one("#max-tokens", Input).value = str(cfg.get("max_tokens", 1024))
        self.query_one("#system-prompt", Input).value = str(cfg.get("system_prompt", "") or "")

    def _save_model_config(self) -> None:
        config = self._model_config_from_form()
        payload = asdict(config)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.notify(f"Saved model config: {self.config_file}", title="Saved")

    def _set_stat_card(self, card_id: str, title: str, value: int) -> None:
        self.query_one(card_id, Static).update(f"{title}\n{value}")

    def _reset_run_stats(self) -> None:
        self._run_completed = 0
        self._run_passed = 0
        self._run_failed = 0
        self._set_stat_card("#card-completed", "Completed", 0)
        self._set_stat_card("#card-passed", "Passed", 0)
        self._set_stat_card("#card-failed", "Failed", 0)

    def _estimate_total_records(self, selected: list[str], limit: int | None) -> int:
        total = 0
        for task in selected:
            try:
                info = self.mapper.get_dataset_info(task, "python")
                count = int(info["record_count"])
                total += min(count, limit) if limit else count
            except Exception:
                continue
        return total

    def _safe_output_dir(self) -> Path:
        raw = self.query_one("#output-dir", Input).value.strip() or "reports/outputs/python"
        path = Path(raw)
        if not path.is_absolute():
            path = (self.root_dir / path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _on_progress_event(self, task_name: str, record_id: str, status: BenchmarkStatus) -> None:
        self._event_queue.put(
            {
                "type": "progress",
                "task": task_name,
                "record": str(record_id),
                "status": status.value if isinstance(status, BenchmarkStatus) else str(status),
                "at": datetime.now().strftime("%H:%M:%S"),
            }
        )

    def _run_suite_worker(
        self,
        model_cfg: ModelConfig,
        selected: list[str],
        limit: int | None,
        run_name: str,
        target_language: str,
        output_dir: Path,
    ) -> None:
        try:
            provider = ProviderRegistry.create(model_cfg)
            suite_cfg = SuiteConfig(
                name=run_name,
                selected_benchmarks=selected,
                language="python",
                output_dir=str(output_dir),
                progress_callback=self._on_progress_event,
                target_language=target_language or None,
            )
            suite = TestSuite(provider, suite_cfg)
            suite.register_benchmarks()

            if limit:
                original_loader = suite._mapper.load_dataset

                def limited_loader(task_name: str, language: str):
                    return original_loader(task_name, language, limit=limit)

                suite._mapper.load_dataset = limited_loader  # type: ignore[assignment]

            results = suite.run_all()
            summary = suite.get_summary()
            save_path = self._save_run_results(
                output_dir=output_dir,
                run_name=run_name,
                model_cfg=model_cfg,
                selected=selected,
                limit=limit,
                summary=summary,
                results=results,
            )

            self._event_queue.put(
                {
                    "type": "done",
                    "summary": summary,
                    "save_path": str(save_path),
                }
            )
        except Exception as exc:
            self._event_queue.put({"type": "error", "message": str(exc)})

    def _save_run_results(
        self,
        output_dir: Path,
        run_name: str,
        model_cfg: ModelConfig,
        selected: list[str],
        limit: int | None,
        summary: dict[str, Any],
        results: list[Any],
    ) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_name).strip("_") or "run"
        file_path = output_dir / f"{stamp}_{slug}.json"

        payload = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "language": "python",
            "selected_benchmarks": selected,
            "dataset_limit": limit,
            "model_config": to_jsonable(model_cfg),
            "summary": to_jsonable(summary),
            "results": [to_jsonable(result) for result in results],
        }
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return file_path

    def _append_log(self, message: str) -> None:
        self.query_one("#run-log", RichLog).write(message)

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except Empty:
                break

            etype = event.get("type")
            if etype == "progress":
                self._run_completed += 1
                status = str(event["status"])
                if status == BenchmarkStatus.PASSED.value:
                    self._run_passed += 1
                elif status in (BenchmarkStatus.FAILED.value, BenchmarkStatus.ERROR.value):
                    self._run_failed += 1

                self._set_stat_card("#card-completed", "Completed", self._run_completed)
                self._set_stat_card("#card-passed", "Passed", self._run_passed)
                self._set_stat_card("#card-failed", "Failed", self._run_failed)

                self.query_one("#run-table", DataTable).add_row(
                    event["task"],
                    event["record"],
                    status,
                    event["at"],
                )
                self._append_log(f"[{event['at']}] {event['task']} #{event['record']} -> {status}")

            elif etype == "done":
                self._running = False
                self.query_one("#start-run", Button).disabled = False
                summary = event["summary"]
                self.query_one("#run-status", Static).update(
                    f"Finished | pass rate {summary.get('pass_rate', 0):.1%} | elapsed {summary.get('elapsed_s', 0):.1f}s"
                )
                self._append_log(f"Saved run file: {event['save_path']}")
                self.notify(f"Run finished. Saved to {event['save_path']}", title="Run Complete")
                self.query_one("#results-dir", Input).value = self.query_one("#output-dir", Input).value
                self._refresh_results()

            elif etype == "error":
                self._running = False
                self.query_one("#start-run", Button).disabled = False
                self.query_one("#run-status", Static).update("Run failed")
                self._append_log(f"ERROR: {event['message']}")
                self.notify(event["message"], title="Run Error", severity="error")

    def _refresh_results(self) -> None:
        raw_dir = self.query_one("#results-dir", Input).value.strip() or "reports/outputs/python"
        base_dir = Path(raw_dir)
        if not base_dir.is_absolute():
            base_dir = (self.root_dir / base_dir).resolve()

        self._report_paths = []
        if base_dir.exists():
            self._report_paths = sorted(base_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        table = self.query_one("#results-table", DataTable)
        table.clear()
        for idx, path in enumerate(self._report_paths):
            stat = path.stat()
            table.add_row(
                str(path.relative_to(self.root_dir)) if path.is_relative_to(self.root_dir) else str(path),
                datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                f"{stat.st_size / 1024:.1f}",
                key=str(idx),
            )

        if not self._report_paths:
            self.query_one("#results-preview", Static).update("No saved run files found in this folder.")

    def _preview_saved_file(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            text = json.dumps(json.loads(text), indent=2)
        except json.JSONDecodeError:
            pass
        return text[:7000] + ("\n\n... truncated ..." if len(text) > 7000 else "")

    def _start_run(self) -> None:
        if self._running:
            self.notify("A run is already active.", severity="warning")
            return

        model_cfg = self._model_config_from_form()
        if not model_cfg.model_name:
            self.notify("Model name is required.", severity="error")
            return

        selected = self._selected_benchmarks()
        if not selected:
            self.notify("Select at least one benchmark.", severity="error")
            return

        limit = self._dataset_limit()
        run_name = self.query_one("#run-name", Input).value.strip() or "Python benchmark run"
        target_language = self.query_one("#target-language", Input).value.strip()
        output_dir = self._safe_output_dir()

        run_table = self.query_one("#run-table", DataTable)
        run_table.clear()
        self.query_one("#run-log", RichLog).clear()

        self._run_total = self._estimate_total_records(selected, limit)
        self._set_stat_card("#card-total", "Total", self._run_total)
        self._reset_run_stats()

        self.query_one("#run-status", Static).update("Running")
        self.query_one("#start-run", Button).disabled = True
        self._append_log(f"Starting run '{run_name}' for {len(selected)} benchmark(s)")
        self._append_log(f"Output folder: {output_dir}")
        self._running = True

        self._run_thread = Thread(
            target=self._run_suite_worker,
            kwargs={
                "model_cfg": model_cfg,
                "selected": selected,
                "limit": limit,
                "run_name": run_name,
                "target_language": target_language,
                "output_dir": output_dir,
            },
            daemon=True,
        )
        self._run_thread.start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "load-config":
            self._load_model_config_into_form()
            self.notify("Loaded model config.")
        elif button_id == "save-config":
            self._save_model_config()
        elif button_id == "start-run":
            self._start_run()
        elif button_id == "refresh-results":
            self._refresh_results()
            self.notify("Results refreshed.")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "results-table":
            return
        row_key = event.row_key.value if event.row_key is not None else None
        if row_key is None:
            return
        idx = int(row_key)
        if 0 <= idx < len(self._report_paths):
            self.query_one("#results-preview", Static).update(self._preview_saved_file(self._report_paths[idx]))

    def action_start_run(self) -> None:
        self._start_run()

    def action_clear_log(self) -> None:
        self.query_one("#run-log", RichLog).clear()

    def action_save_model_config(self) -> None:
        self._save_model_config()

    def action_refresh_results(self) -> None:
        self._refresh_results()


if __name__ == "__main__":
    BenchmarkTUI().run()

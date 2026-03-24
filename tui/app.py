from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)


@dataclass
class DashboardSnapshot:
    providers: int
    benchmarks: int
    dimensions: int
    dataset_files: int
    report_files: int


class CodebaseScanner:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def providers_dir(self) -> Path:
        return self.root / "providers"

    @property
    def tasks_dir(self) -> Path:
        return self.root / "benchmarks" / "tasks"

    @property
    def dimensions_dir(self) -> Path:
        return self.root / "benchmarks" / "dimensions"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def provider_names(self) -> list[str]:
        if not self.providers_dir.exists():
            return []
        names: list[str] = []
        for child in self.providers_dir.iterdir():
            if child.is_dir() and (child / "provider.py").exists():
                names.append(child.name)
        return sorted(names)

    @staticmethod
    def _python_modules(path: Path) -> list[str]:
        if not path.exists():
            return []
        modules: list[str] = []
        for file in path.glob("*.py"):
            if file.name.startswith("__"):
                continue
            modules.append(file.stem)
        return sorted(modules)

    def benchmark_names(self) -> list[str]:
        return self._python_modules(self.tasks_dir)

    def dimension_names(self) -> list[str]:
        return self._python_modules(self.dimensions_dir)

    def dataset_files(self) -> list[Path]:
        if not self.datasets_dir.exists():
            return []
        return sorted(
            [p for p in self.datasets_dir.rglob("*.csv") if p.is_file()],
            key=lambda p: str(p).lower(),
        )

    def report_files(self) -> list[Path]:
        if not self.reports_dir.exists():
            return []
        exts = {".json", ".csv", ".md", ".txt"}
        return sorted(
            [p for p in self.reports_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def snapshot(self) -> DashboardSnapshot:
        return DashboardSnapshot(
            providers=len(self.provider_names()),
            benchmarks=len(self.benchmark_names()),
            dimensions=len(self.dimension_names()),
            dataset_files=len(self.dataset_files()),
            report_files=len(self.report_files()),
        )


class BenchmarkTUI(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "LLM Benchmark TUI"
    SUB_TITLE = "Dashboard, Model Config, Previous Results"

    BINDINGS = [
        ("d", "switch_tab('dashboard')", "Dashboard"),
        ("m", "switch_tab('model-config')", "Model Config"),
        ("r", "switch_tab('results')", "Results"),
        ("ctrl+r", "refresh_all", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.root_dir = Path(__file__).resolve().parent.parent
        self.scanner = CodebaseScanner(self.root_dir)
        self.config_file = self.root_dir / "config" / "model_config.yaml"
        self._report_paths: list[Path] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="main-tabs", initial="dashboard"):
            with TabPane("Main Dashboard", id="dashboard"):
                with VerticalScroll(classes="pane-root"):
                    with Horizontal(classes="card-row"):
                        yield Static("Providers\n0", id="card-providers", classes="card")
                        yield Static("Benchmarks\n0", id="card-benchmarks", classes="card")
                        yield Static("Dimensions\n0", id="card-dimensions", classes="card")
                        yield Static("Datasets\n0", id="card-datasets", classes="card")
                        yield Static("Reports\n0", id="card-reports", classes="card")
                    yield Static("Task x Dimension Matrix", classes="section-title")
                    yield DataTable(id="benchmark-table")
                    with Horizontal(id="dashboard-actions"):
                        yield Button("Refresh Dashboard", id="refresh-dashboard", variant="primary")

            with TabPane("Model Config", id="model-config"):
                with VerticalScroll(classes="pane-root"):
                    with Horizontal(classes="form-row"):
                        yield Label("Provider", classes="form-label")
                        yield Select([], id="provider", classes="form-field")
                    with Horizontal(classes="form-row"):
                        yield Label("Model Name", classes="form-label")
                        yield Input(placeholder="e.g. gpt-4o-mini", id="model-name", classes="form-field")
                    with Horizontal(classes="form-row"):
                        yield Label("Base URL", classes="form-label")
                        yield Input(placeholder="Optional endpoint", id="base-url", classes="form-field")
                    with Horizontal(classes="form-row"):
                        yield Label("API Key", classes="form-label")
                        yield Input(placeholder="Optional secret", password=True, id="api-key", classes="form-field")
                    with Horizontal(classes="form-row"):
                        yield Label("Temperature", classes="form-label")
                        yield Input(value="0.7", id="temperature", classes="form-field")
                    with Horizontal(classes="form-row"):
                        yield Label("Max Tokens", classes="form-label")
                        yield Input(value="4096", id="max-tokens", classes="form-field")
                    with Horizontal(classes="form-row"):
                        yield Label("System Prompt", classes="form-label")
                        yield Input(placeholder="Optional system instruction", id="system-prompt", classes="form-field")
                    with Horizontal(id="model-actions"):
                        yield Button("Load Config", id="load-config")
                        yield Button("Save Config", id="save-config", variant="success")

            with TabPane("Previous Results", id="results"):
                with VerticalScroll(classes="pane-root"):
                    yield Static("Saved Report Files", classes="section-title")
                    yield DataTable(id="results-table")
                    yield Static("Select a report row to preview file contents.", id="results-preview")
                    yield Button("Refresh Results", id="refresh-results")

        yield Footer()

    def on_mount(self) -> None:
        self._init_tables()
        self._refresh_dashboard()
        self._load_model_config_into_form()
        self._refresh_results()

    def _init_tables(self) -> None:
        benchmark_table = self.query_one("#benchmark-table", DataTable)
        benchmark_table.cursor_type = "row"
        benchmark_table.add_columns("Benchmark", "Dimension Count")

        results_table = self.query_one("#results-table", DataTable)
        results_table.cursor_type = "row"
        results_table.add_columns("File", "Type", "Modified", "Size (KB)")

    def _safe_set_card(self, card_id: str, title: str, value: int) -> None:
        self.query_one(card_id, Static).update(f"{title}\\n{value}")

    def _refresh_dashboard(self) -> None:
        snapshot = self.scanner.snapshot()
        self._safe_set_card("#card-providers", "Providers", snapshot.providers)
        self._safe_set_card("#card-benchmarks", "Benchmarks", snapshot.benchmarks)
        self._safe_set_card("#card-dimensions", "Dimensions", snapshot.dimensions)
        self._safe_set_card("#card-datasets", "Datasets", snapshot.dataset_files)
        self._safe_set_card("#card-reports", "Reports", snapshot.report_files)

        benchmark_table = self.query_one("#benchmark-table", DataTable)
        benchmark_table.clear()

        dims_per_task = self._task_dimension_count()
        for benchmark_name in self.scanner.benchmark_names():
            benchmark_table.add_row(benchmark_name, str(dims_per_task.get(benchmark_name, 0)))

    def _task_dimension_count(self) -> dict[str, int]:
        try:
            from benchmarks.matrix import BENCHMARK_MATRIX

            return {task: len(dimensions) for task, dimensions in BENCHMARK_MATRIX.items()}
        except Exception:
            return {}

    def _provider_options(self) -> list[tuple[str, str]]:
        providers = self.scanner.provider_names()
        if not providers:
            return [("none", "none")]
        return [(p, p) for p in providers]

    def _load_config_file(self) -> dict:
        if not self.config_file.exists():
            return {}
        try:
            return yaml.safe_load(self.config_file.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _load_model_config_into_form(self) -> None:
        provider_widget = self.query_one("#provider", Select)
        provider_widget.set_options(self._provider_options())

        config = self._load_config_file()
        provider = str(config.get("provider", "ollama"))
        if provider not in {value for _, value in self._provider_options()}:
            provider = self._provider_options()[0][1]

        provider_widget.value = provider
        self.query_one("#model-name", Input).value = str(config.get("model_name", ""))
        self.query_one("#base-url", Input).value = str(config.get("base_url", ""))
        self.query_one("#api-key", Input).value = str(config.get("api_key", ""))
        self.query_one("#temperature", Input).value = str(config.get("temperature", 0.7))
        self.query_one("#max-tokens", Input).value = str(config.get("max_tokens", 4096))
        self.query_one("#system-prompt", Input).value = str(config.get("system_prompt", ""))

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

    def _save_model_config(self) -> None:
        provider = self.query_one("#provider", Select).value
        payload = {
            "provider": provider,
            "model_name": self.query_one("#model-name", Input).value.strip(),
            "base_url": self.query_one("#base-url", Input).value.strip() or None,
            "api_key": self.query_one("#api-key", Input).value.strip() or None,
            "temperature": self._parse_float(self.query_one("#temperature", Input).value.strip(), 0.7),
            "max_tokens": self._parse_int(self.query_one("#max-tokens", Input).value.strip(), 4096),
            "system_prompt": self.query_one("#system-prompt", Input).value.strip() or None,
            "extra_params": {},
        }

        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        self.notify(f"Saved model config to {self.config_file}", title="Config Saved")

    def _refresh_results(self) -> None:
        self._report_paths = self.scanner.report_files()
        table = self.query_one("#results-table", DataTable)
        table.clear()

        for index, path in enumerate(self._report_paths):
            stat = path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = f"{stat.st_size / 1024:.1f}"
            table.add_row(
                str(path.relative_to(self.root_dir)),
                path.suffix.lower().lstrip("."),
                modified,
                size_kb,
                key=str(index),
            )

        preview = self.query_one("#results-preview", Static)
        if not self._report_paths:
            preview.update("No report files found in reports/. Run benchmarks to populate this tab.")

    def _preview_file(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")

        if path.suffix.lower() == ".json":
            try:
                parsed = json.loads(text)
                text = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass

        limit = 5000
        if len(text) > limit:
            return text[:limit] + "\n\n... truncated ..."
        return text

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "refresh-dashboard":
            self._refresh_dashboard()
            self.notify("Dashboard refreshed.")
        elif button_id == "load-config":
            self._load_model_config_into_form()
            self.notify("Model config loaded.")
        elif button_id == "save-config":
            self._save_model_config()
        elif button_id == "refresh-results":
            self._refresh_results()
            self.notify("Results list refreshed.")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "results-table":
            return

        row_key = event.row_key.value if event.row_key is not None else None
        if row_key is None:
            return

        index = int(row_key)
        if index < 0 or index >= len(self._report_paths):
            return

        path = self._report_paths[index]
        content = self._preview_file(path)
        self.query_one("#results-preview", Static).update(content)

    def action_refresh_all(self) -> None:
        self._refresh_dashboard()
        self._load_model_config_into_form()
        self._refresh_results()
        self.notify("All views refreshed.")

    def action_switch_tab(self, tab_id: str) -> None:
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.active = tab_id


if __name__ == "__main__":
    BenchmarkTUI().run()

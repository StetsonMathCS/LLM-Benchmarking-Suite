"""
run_experiment.py
Headless CLI runner — bypasses the TUI entirely.

Usage:
    python run_experiment.py                  # interactive prompts
    python run_experiment.py --config config/model_config.yaml  # load config, still prompt
    python run_experiment.py --yes            # use config file values without prompting
    python run_experiment.py --yes --tasks bug_fixing refactoring
    python run_experiment.py --help

Run from the codebase root directory.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_TASKS = [
    "bug_fixing",
    "code_generation",
    "code_review",
    "refactoring",
    "test_generation",
    "translation",
]

PROVIDERS = ["openai", "anthropic", "ollama", "huggingface"]

# ANSI colours (stripped when not a tty)
def _c(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def bold(t):   return _c("1", t)
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def cyan(t):   return _c("36", t)
def red(t):    return _c("31", t)
def dim(t):    return _c("2", t)


def _prompt(label: str, default: Optional[str] = None, secret: bool = False) -> str:
    """Prompt user for a value; returns default if user presses Enter."""
    hint = f" [{dim(str(default))}]" if default not in (None, "") else ""
    suffix = ": "
    if secret:
        import getpass
        value = getpass.getpass(f"  {label}{hint}{suffix}")
    else:
        value = input(f"  {label}{hint}{suffix}").strip()
    return value if value else (default or "")


def _prompt_int(label: str, default: Optional[int] = None) -> Optional[int]:
    raw = _prompt(label, str(default) if default is not None else None)
    if raw in ("", "none", "None", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        print(yellow(f"  Invalid integer '{raw}', using default {default}"))
        return default


def _prompt_float(label: str, default: float) -> float:
    raw = _prompt(label, str(default))
    if raw in ("", "none", "None"):
        return default
    try:
        return float(raw)
    except ValueError:
        print(yellow(f"  Invalid float '{raw}', using default {default}"))
        return default


def _section(title: str):
    width = 60
    print()
    print(bold(cyan("─" * width)))
    print(bold(cyan(f"  {title}")))
    print(bold(cyan("─" * width)))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(yellow(f"Config file not found: {path}"))
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    print(green(f"Loaded config from {path}"))
    return data


# ---------------------------------------------------------------------------
# Interactive config collection
# ---------------------------------------------------------------------------

def collect_model_config(defaults: dict, yes: bool) -> dict:
    """Collect model config, optionally prompting user for each field."""
    _section("Model Configuration")

    if yes:
        cfg = {
            "provider":      defaults.get("provider", "openai"),
            "model_name":    defaults.get("model_name", ""),
            "api_key":       defaults.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "",
            "base_url":      defaults.get("base_url") or "",
            "temperature":   defaults.get("temperature", 0.7),
            "max_tokens":    defaults.get("max_tokens", 1024),
            "system_prompt": defaults.get("system_prompt") or "",
        }
        # Print what we're using
        for k, v in cfg.items():
            display = "***" if k == "api_key" and v else (v or dim("(not set)"))
            print(f"  {k}: {display}")
        return cfg

    provider = _prompt("Provider (openai/anthropic/ollama/huggingface)",
                       defaults.get("provider", "openai"))
    model_name = _prompt("Model name", defaults.get("model_name", ""))

    # API key: try env var first as a smart default
    env_key = ""
    if provider == "openai":
        env_key = os.environ.get("OPENAI_API_KEY", "")
    elif provider == "anthropic":
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    key_default = defaults.get("api_key") or env_key or ""
    api_key = _prompt("API key (leave blank if not needed)", key_default, secret=True)

    base_url   = _prompt("Base URL (leave blank for default)", defaults.get("base_url") or "")
    temperature = _prompt_float("Temperature", float(defaults.get("temperature", 0.7)))
    max_tokens  = _prompt_int("Max tokens", int(defaults.get("max_tokens", 1024)))
    system_prompt = _prompt("System prompt (optional)", defaults.get("system_prompt") or "")

    return {
        "provider":      provider,
        "model_name":    model_name,
        "api_key":       api_key or None,
        "base_url":      base_url or None,
        "temperature":   temperature,
        "max_tokens":    max_tokens,
        "system_prompt": system_prompt or None,
    }


def collect_run_config(defaults: dict, yes: bool, cli_tasks: Optional[list]) -> dict:
    """Collect run/suite config."""
    _section("Run Configuration")

    if yes:
        cfg = {
            "run_name":       defaults.get("run_name", "headless_run"),
            "language":       defaults.get("language", "python"),
            "target_language": defaults.get("target_language", "javascript"),
            "output_dir":     defaults.get("output_dir", "reports/outputs/python"),
            "dataset_limit":  defaults.get("dataset_limit") or None,
            "pass_threshold": float(defaults.get("pass_threshold", 0.5)),
            "num_samples":    int(defaults.get("num_samples", 1)),
            "pass_k_values":  defaults.get("pass_k_values", [1, 5, 10]),
        }
        for k, v in cfg.items():
            print(f"  {k}: {v if v is not None else dim('(not set)')}")
        return cfg

    run_name = _prompt("Run name", defaults.get("run_name", "headless_run"))
    language = _prompt("Language (python/cpp/javascript)", defaults.get("language", "python"))
    target_language = _prompt("Translation target language", defaults.get("target_language", "javascript"))
    output_dir = _prompt("Output directory", defaults.get("output_dir", "reports/outputs/python"))
    dataset_limit = _prompt_int("Dataset row limit per task (blank = all)", defaults.get("dataset_limit"))
    pass_threshold = _prompt_float("Pass threshold (0.0–1.0)", float(defaults.get("pass_threshold", 0.5)))
    num_samples = _prompt_int("Samples per problem for pass@k (1 = off)", defaults.get("num_samples", 1)) or 1
    pass_k_input = _prompt("pass@k k-values (space-separated)", " ".join(str(k) for k in defaults.get("pass_k_values", [1, 5, 10])))
    pass_k_values = [int(x) for x in pass_k_input.split() if x.isdigit()] or [1, 5, 10]

    return {
        "run_name":        run_name,
        "language":        language,
        "target_language": target_language,
        "output_dir":      output_dir,
        "dataset_limit":   dataset_limit,
        "pass_threshold":  pass_threshold,
        "num_samples":     num_samples,
        "pass_k_values":   pass_k_values,
    }


def collect_task_selection(yes: bool, cli_tasks: Optional[list]) -> list:
    """Return list of selected task names."""
    _section("Task Selection")
    print(f"  Available tasks:")
    for i, t in enumerate(ALL_TASKS, 1):
        print(f"    {dim(str(i))}) {t}")

    if cli_tasks:
        selected = [t for t in cli_tasks if t in ALL_TASKS]
        unknown = [t for t in cli_tasks if t not in ALL_TASKS]
        if unknown:
            print(yellow(f"  Unknown tasks ignored: {unknown}"))
        print(f"  Selected (from --tasks): {selected}")
        return selected

    if yes:
        print(f"  Selected (all): {ALL_TASKS}")
        return list(ALL_TASKS)

    print()
    print("  Enter task numbers separated by spaces, or press Enter to select all.")
    print(f"  Example: {dim('1 3 6')}")
    raw = input("  Selection: ").strip()
    if not raw:
        print(f"  Selected: all")
        return list(ALL_TASKS)

    selected = []
    for part in raw.split():
        try:
            idx = int(part) - 1
            if 0 <= idx < len(ALL_TASKS):
                selected.append(ALL_TASKS[idx])
        except ValueError:
            if part in ALL_TASKS:
                selected.append(part)
    if not selected:
        print(yellow("  No valid tasks selected, defaulting to all."))
        return list(ALL_TASKS)
    print(f"  Selected: {selected}")
    return selected


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self):
        self.total = 0
        self.done = 0
        self.passed = 0
        self.failed = 0
        self._start = time.perf_counter()

    def callback(self, task_name: str, record_id: str, status, score: Optional[float]):
        from core.base import BenchmarkStatus
        self.done += 1
        if status == BenchmarkStatus.PASSED:
            self.passed += 1
            icon = green("✓")
        elif status == BenchmarkStatus.ERROR:
            self.failed += 1
            icon = red("✗")
        else:
            self.failed += 1
            icon = yellow("✗")

        score_str = f"{score:.3f}" if score is not None else "  N/A"
        elapsed = time.perf_counter() - self._start
        print(f"  {icon}  [{elapsed:6.1f}s]  {task_name:<22}  record {str(record_id):<14}  score={score_str}")


# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------

def save_report(summary: dict, model_cfg: dict, run_cfg: dict, output_dir: str) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_cfg.get("run_name", "run").replace(" ", "_")
    filename = out / f"{ts}_{name}.json"

    # Strip raw results to keep file reasonable; include full summary
    metadata = {
        "run_name": run_cfg.get("run_name"),
        "timestamp": ts,
        "language": run_cfg.get("language"),
    }
    num_samples = run_cfg.get("num_samples", 1)
    if num_samples > 1:
        metadata["num_samples"] = num_samples
        metadata["pass_k_values"] = run_cfg.get("pass_k_values", [1, 5, 10])

    report = {
        "metadata": metadata,
        "model_config": {k: v for k, v in model_cfg.items() if k != "api_key"},
        "summary": {k: v for k, v in summary.items() if k != "results"},
        "results": summary.get("results", []),
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return str(filename)


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(summary: dict):
    _section("Results Summary")
    score = summary.get("final_score", 0.0)
    grade = summary.get("grade", "?")
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    errors = summary.get("errors", 0)
    elapsed = summary.get("elapsed_s", 0)

    print(f"  Score  : {bold(f'{score:.4f}')}")
    print(f"  Grade  : {bold(grade)}")
    print(f"  Total  : {total}  |  Passed: {green(str(passed))}  |  Failed: {yellow(str(failed))}  |  Errors: {red(str(errors))}")
    print(f"  Elapsed: {elapsed:.1f}s")

    task_scores = summary.get("task_scores", {})
    pass_at_k = summary.get("pass_at_k", {})

    if task_scores:
        print()
        print(f"  {'Task':<24}  {'Score':>7}  {'Pass Rate':>9}  {'Records':>7}")
        print(f"  {'-'*24}  {'-'*7}  {'-'*9}  {'-'*7}")
        for task, ts_dict in task_scores.items():
            ms = ts_dict.get("mean_score", 0.0)
            pr = ts_dict.get("pass_rate", 0.0)
            rc = ts_dict.get("record_count", 0)
            bar = "█" * int(ms * 10) + "░" * (10 - int(ms * 10))
            print(f"  {task:<24}  {ms:>7.4f}  {pr:>8.1%}  {rc:>7}  {dim(bar)}")

    # pass@k table (only shown in multi-sample mode)
    if pass_at_k:
        # Collect all k values across tasks
        all_k = sorted({k for kv in pass_at_k.values() for k in kv})
        if all_k:
            _section("pass@k Results")
            k_headers = "  ".join(f"{'pass@'+str(k):>8}" for k in all_k)
            print(f"  {'Task':<24}  {k_headers}")
            print(f"  {'-'*24}  {'  '.join('-'*8 for _ in all_k)}")
            for task in task_scores:
                task_pak = pass_at_k.get(task, {})
                vals = "  ".join(
                    f"{task_pak.get(k, float('nan')):>8.4f}"
                    if k in task_pak else f"{'N/A':>8}"
                    for k in all_k
                )
                print(f"  {task:<24}  {vals}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Headless benchmark runner — no TUI required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py
  python run_experiment.py --config config/model_config.yaml
  python run_experiment.py --yes --tasks bug_fixing refactoring
  python run_experiment.py --yes --limit 5
        """,
    )
    parser.add_argument("--config", default="config/model_config.yaml",
                        help="Path to YAML config file (default: config/model_config.yaml)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Non-interactive: use config file values, skip all prompts")
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, metavar="TASK",
                        help=f"Tasks to run. Choices: {', '.join(ALL_TASKS)}")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max dataset rows per task")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory for reports")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Samples per problem for pass@k estimation (default: 1, no pass@k)")
    parser.add_argument("--pass-k", nargs="+", type=int, default=None,
                        help="k values for pass@k (default: 1 5 10)")
    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print(bold(cyan("╔══════════════════════════════════════════════════════╗")))
    print(bold(cyan("║         LLM Benchmark Suite — Headless Runner        ║")))
    print(bold(cyan("╚══════════════════════════════════════════════════════╝")))

    # --- Prompt for config path (useful when running multiple instances in tmux) ---
    config_path = args.config
    if not args.yes:
        user_path = input(f"\n  Config file path [{dim(config_path)}]: ").strip()
        if user_path:
            config_path = user_path

    # --- Load YAML config ---
    file_defaults = load_yaml_config(config_path)

    # CLI args override file defaults where specified
    if args.limit is not None:
        file_defaults["dataset_limit"] = args.limit
    if args.output_dir:
        file_defaults["output_dir"] = args.output_dir
    if args.num_samples is not None:
        file_defaults["num_samples"] = args.num_samples
    if args.pass_k is not None:
        file_defaults["pass_k_values"] = args.pass_k

    # --- Collect configs ---
    model_cfg  = collect_model_config(file_defaults, args.yes)
    run_cfg    = collect_run_config(file_defaults, args.yes, args.tasks)
    tasks      = collect_task_selection(args.yes, args.tasks)

    if not tasks:
        print(red("No tasks selected. Exiting."))
        sys.exit(1)

    if not model_cfg.get("model_name"):
        print(red("model_name is required. Exiting."))
        sys.exit(1)

    # --- Confirm ---
    _section("Ready to Run")
    print(f"  Provider : {model_cfg['provider']}")
    print(f"  Model    : {model_cfg['model_name']}")
    print(f"  Tasks    : {', '.join(tasks)}")
    print(f"  Language : {run_cfg['language']}")
    limit_display = str(run_cfg.get("dataset_limit")) if run_cfg.get("dataset_limit") else "all rows"
    print(f"  Limit    : {limit_display} per task")
    num_samples = run_cfg.get("num_samples", 1)
    if num_samples > 1:
        k_vals = run_cfg.get("pass_k_values", [1, 5, 10])
        print(f"  Samples  : {num_samples} per problem  (pass@k for k={k_vals})")
    print(f"  Output   : {run_cfg['output_dir']}")

    if not args.yes:
        print()
        confirm = input("  Start? [Y/n]: ").strip().lower()
        if confirm == "n":
            print("Aborted.")
            sys.exit(0)

    # --- Build core objects ---
    from core.base import ModelConfig
    from core.registry import ProviderRegistry
    from core.suite import TestSuite, SuiteConfig

    mc = ModelConfig(
        provider=model_cfg["provider"],
        model_name=model_cfg["model_name"],
        api_key=model_cfg.get("api_key") or None,
        base_url=model_cfg.get("base_url") or None,
        temperature=model_cfg.get("temperature"),
        max_tokens=model_cfg.get("max_tokens"),
        system_prompt=model_cfg.get("system_prompt") or None,
    )

    _section("Connecting to Provider")
    print(f"  Connecting to {mc.provider} / {mc.model_name} ...")
    try:
        provider = ProviderRegistry.create(mc)
        print(green("  Connected."))
    except Exception as e:
        print(red(f"  Failed to connect: {e}"))
        sys.exit(1)

    tracker = ProgressTracker()

    sc = SuiteConfig(
        name=run_cfg["run_name"],
        selected_benchmarks=tasks,
        language=run_cfg["language"],
        output_dir=run_cfg["output_dir"],
        progress_callback=tracker.callback,
        target_language=run_cfg.get("target_language") or "javascript",
        pass_threshold=run_cfg.get("pass_threshold"),
        num_samples=run_cfg.get("num_samples", 1),
        pass_k_values=run_cfg.get("pass_k_values", [1, 5, 10]),
    )

    suite = TestSuite(provider, sc)
    suite.register_benchmarks()

    # Apply dataset_limit by patching the mapper's load_dataset
    dataset_limit = run_cfg.get("dataset_limit")
    if dataset_limit:
        original_load = suite._mapper.load_dataset

        def _limited_load(task_name, language, limit=None):
            return original_load(task_name, language, limit=dataset_limit)

        suite._mapper.load_dataset = _limited_load

    # --- Run ---
    _section(f"Running Benchmarks  ({len(tasks)} task(s))")
    print(f"  {'Status':<4}  {'Elapsed':>8}  {'Task':<22}  {'Record':<10}  {'Score'}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*22}  {'-'*10}  {'-'*7}")

    try:
        suite.run_all()
    except KeyboardInterrupt:
        print()
        print(yellow("  Interrupted by user — computing partial results..."))

    summary = suite.get_summary()

    # --- Print results ---
    print_summary(summary)

    # --- Save report ---
    _section("Saving Report")
    try:
        report_path = save_report(summary, model_cfg, run_cfg, run_cfg["output_dir"])
        print(f"  Saved: {green(report_path)}")
    except Exception as e:
        print(red(f"  Could not save report: {e}"))

    print()
    print(bold(green("Done.")))
    print()


if __name__ == "__main__":
    main()

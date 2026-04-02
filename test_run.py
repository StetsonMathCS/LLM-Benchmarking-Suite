"""
test_run.py
Smoke-test runner: 5 records per task, all 8 tasks.
Loads model config from config/model_config.yaml.

Usage:
    python test_run.py
    python test_run.py --config config/model_config.yaml
    python test_run.py --tasks bug_fixing refactoring   # subset only
    python test_run.py --no-dim                         # hide per-dimension breakdown
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

def bold(t):  return _c("1", t)
def green(t): return _c("32", t)
def red(t):   return _c("31", t)
def yellow(t):return _c("33", t)
def cyan(t):  return _c("36", t)
def dim(t):   return _c("2", t)
def magenta(t): return _c("35", t)

LIMIT = 5

ALL_TASKS = [
    "bug_fixing",
    "code_completion",
    "code_generation",
    "code_review",
    "partial_transform",
    "refactoring",
    "test_generation",
    "translation",
]

# ---------------------------------------------------------------------------
# Dimension logging
# ---------------------------------------------------------------------------

def _serialize_dim_details(details: dict) -> dict:
    """Recursively make DimensionResult details JSON-safe."""
    out = {}
    for k, v in details.items():
        if hasattr(v, "__dict__"):
            out[k] = v.__dict__
        elif isinstance(v, dict):
            out[k] = _serialize_dim_details(v)
        else:
            out[k] = str(v) if not isinstance(v, (int, float, bool, type(None), list)) else v
    return out


def print_dim_log(task_name: str, record_id: str, br, weights: dict):
    """Print a per-dimension breakdown table for one BenchmarkResult."""
    from core.base import DimensionResult

    details = br.details  # {dim_name: DimensionResult}
    if not details:
        print(f"           {dim('(no dimension details)')}")
        return

    print(f"           {'Dimension':<30}  {'Weight':>6}  {'Score':>6}  {'Pass':>5}  Issues")
    print(f"           {'-'*30}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*30}")

    for dim_name, dr in details.items():
        if not isinstance(dr, DimensionResult):
            # Fallback: might be stored as plain dict on error paths
            score  = dr.get("score", 0.0) if isinstance(dr, dict) else 0.0
            passed = dr.get("passed", False) if isinstance(dr, dict) else False
            issues_list = dr.get("issues", []) if isinstance(dr, dict) else []
            extra_details = dr.get("details", {}) if isinstance(dr, dict) else {}
        else:
            score       = dr.score
            passed      = dr.passed
            issues_list = dr.issues
            extra_details = dr.details

        weight     = weights.get(dim_name, 0.0)
        pass_icon  = green("✓") if passed else red("✗")
        score_str  = f"{score:.4f}"
        weight_str = f"{weight:.2f}"

        # Condense issues to one line
        issues_str = ""
        if issues_list:
            first = str(issues_list[0])[:50]
            more  = f" (+{len(issues_list)-1})" if len(issues_list) > 1 else ""
            issues_str = yellow(first + more)
        elif extra_details.get("error"):
            issues_str = red(str(extra_details["error"])[:50])
        else:
            # Show a brief detail hint when available and no issues
            for key in ("note", "reason", "message", "info"):
                if extra_details.get(key):
                    issues_str = dim(str(extra_details[key])[:50])
                    break

        print(f"           {dim_name:<30}  {weight_str:>6}  {score_str:>6}  {pass_icon:>5}  {issues_str}")

    # Extra detail lines: show full issues list for any failing dimension
    failing = {
        name: dr for name, dr in details.items()
        if isinstance(dr, DimensionResult) and (not dr.passed or dr.issues)
    }
    if failing:
        print(f"           {dim('── issue detail ──')}")
        for dim_name, dr in failing.items():
            for issue in dr.issues[:3]:          # cap at 3 per dimension
                print(f"           {yellow('!')} {dim_name}: {str(issue)[:90]}")
            if len(dr.issues) > 3:
                print(f"           {dim(f'  ... and {len(dr.issues)-3} more issues in {dim_name}')}")
            # Show extra_details keys that look informative
            for key in ("output", "generated_output", "expected_output", "diff"):
                val = dr.details.get(key)
                if val:
                    snippet = str(val).replace("\n", " ")[:80]
                    print(f"           {dim(f'  {key}: {snippet}')}")


# ---------------------------------------------------------------------------
# Task-level test
# ---------------------------------------------------------------------------

def test_task(task_name: str, benchmark, mapper, language: str, target_language: str,
              show_dims: bool = True) -> tuple:
    """
    Run up to LIMIT records of one task.
    Returns (stats_dict, list[BenchmarkResult]) — raw results are needed for ScoringEngine.
    """
    from benchmarks.matrix import DIMENSION_WEIGHTS
    from core.base import BenchmarkResult, BenchmarkStatus, DimensionResult

    stats = {
        "task": task_name,
        "records_attempted": 0,
        "records_ok": 0,
        "records_error": 0,
        "scores": [],
        "errors": [],
        "skipped": False,
        "skip_reason": "",
    }
    raw_results: list[BenchmarkResult] = []

    # Load dataset
    try:
        records = mapper.load_dataset(task_name, language, limit=LIMIT)
    except FileNotFoundError as e:
        stats["skipped"] = True
        stats["skip_reason"] = f"Dataset not found: {e}"
        return stats, raw_results
    except Exception as e:
        stats["skipped"] = True
        stats["skip_reason"] = str(e)
        return stats, raw_results

    if not records:
        stats["skipped"] = True
        stats["skip_reason"] = "Dataset returned 0 records"
        return stats, raw_results

    weights = DIMENSION_WEIGHTS.get(task_name, {})

    for rec in records:
        stats["records_attempted"] += 1
        try:
            kwargs = mapper.map_record_to_benchmark_kwargs(rec)
            if task_name == "translation" and target_language:
                kwargs["target_language"] = target_language
            code_input = kwargs.pop("code_input", "")
            br = benchmark._timed_run(code_input=code_input, **kwargs)

            # Tag so ScoringEngine can group by task (same as TestSuite.run_all does)
            br.metadata["task_name"] = task_name
            br.metadata["record_id"] = rec.record_id
            raw_results.append(br)

            score = br.combined_score
            stats["scores"].append(score if score is not None else 0.0)
            stats["records_ok"] += 1

            icon = green("✓") if br.status.value == "passed" else yellow("~")
            score_str = f"{score:.3f}" if score is not None else "  N/A"
            print(f"      {icon}  record {str(rec.record_id):<6}  score={score_str}  [{br.status.value}]  {dim(f'{br.duration_s:.2f}s')}")

            if show_dims:
                print_dim_log(task_name, rec.record_id, br, weights)
                print()

        except Exception as e:
            stats["records_error"] += 1
            stats["errors"].append({"record_id": rec.record_id, "error": str(e)})
            # Still store an error result so ScoringEngine counts it
            err_result = BenchmarkResult(
                benchmark_name=task_name,
                status=BenchmarkStatus.ERROR,
                details={"error": str(e)},
            )
            err_result.metadata["task_name"] = task_name
            err_result.metadata["record_id"] = rec.record_id
            raw_results.append(err_result)
            print(f"      {red('✗')}  record {str(rec.record_id):<6}  ERROR: {str(e)[:80]}")
            if show_dims:
                print(f"           {dim(traceback.format_exc().strip()[:300])}")
                print()

    if stats["scores"]:
        stats["mean_score"] = sum(stats["scores"]) / len(stats["scores"])
    else:
        stats["mean_score"] = None

    return stats, raw_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=f"Smoke test: {LIMIT} records per task.")
    p.add_argument("--config", default="config/model_config.yaml")
    p.add_argument("--tasks", nargs="+", choices=ALL_TASKS, metavar="TASK",
                   help="Subset of tasks to test (default: all)")
    p.add_argument("--no-dim", action="store_true",
                   help="Hide per-dimension breakdown (show record-level only)")
    return p.parse_args()


def main():
    args = parse_args()
    tasks_to_run = args.tasks or ALL_TASKS
    show_dims = not args.no_dim

    # Prompt for config path (useful when running multiple instances in tmux)
    config_path = args.config
    user_path = input(f"\n  Config file path [{dim(config_path)}]: ").strip()
    if user_path:
        config_path = user_path

    # Load config
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(red(f"Config file not found: {config_path}"))
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    provider_name  = cfg.get("provider", "openai")
    model_name     = cfg.get("model_name", "")
    api_key        = cfg.get("api_key") or None
    base_url       = cfg.get("base_url") or None
    temperature    = cfg.get("temperature", 0.7)
    max_tokens     = cfg.get("max_tokens", 1024)
    system_prompt  = cfg.get("system_prompt") or None
    language       = cfg.get("language", "python")
    target_language = cfg.get("target_language", "javascript")

    print()
    print(bold(cyan("╔══════════════════════════════════════════════════════╗")))
    print(bold(cyan(f"║   Smoke Test  —  {LIMIT} records × {len(tasks_to_run)} task(s)                  ║")))
    print(bold(cyan("╚══════════════════════════════════════════════════════╝")))
    print()
    print(f"  Provider  : {provider_name}")
    print(f"  Model     : {model_name}")
    print(f"  Language  : {language}")
    print(f"  Tasks     : {', '.join(tasks_to_run)}")
    print()

    # Connect provider
    from core.base import ModelConfig
    from core.registry import ProviderRegistry
    from datasets.mapper import DatasetMapper

    mc = ModelConfig(
        provider=provider_name,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )

    print(f"  Connecting to {provider_name} / {model_name} ...", end=" ", flush=True)
    try:
        provider = ProviderRegistry.create(mc)
        print(green("OK"))
    except Exception as e:
        print(red(f"FAILED\n  {e}"))
        sys.exit(1)

    mapper = DatasetMapper()

    # Import benchmark classes
    from benchmarks.tasks.bug_fixing import BugFixingBenchmark
    from benchmarks.tasks.code_completion import CodeCompletionBenchmark
    from benchmarks.tasks.code_generation import CodeGenerationBenchmark
    from benchmarks.tasks.code_review import CodeReviewBenchmark
    from benchmarks.tasks.partial_transform import PartialTransformBenchmark
    from benchmarks.tasks.refactoring import RefactoringBenchmark
    from benchmarks.tasks.test_generation import TestGenerationBenchmark
    from benchmarks.tasks.translation import TranslationBenchmark

    benchmark_classes = {
        "bug_fixing":       BugFixingBenchmark,
        "code_completion":  CodeCompletionBenchmark,
        "code_generation":  CodeGenerationBenchmark,
        "code_review":      CodeReviewBenchmark,
        "partial_transform": PartialTransformBenchmark,
        "refactoring":      RefactoringBenchmark,
        "test_generation":  TestGenerationBenchmark,
        "translation":      TranslationBenchmark,
    }

    # -----------------------------------------------------------------------
    # Run each task
    # -----------------------------------------------------------------------
    all_stats = []
    all_raw_results = []   # flat list of BenchmarkResult objects for ScoringEngine
    wall_start = time.perf_counter()

    for task_name in tasks_to_run:
        print()
        print(bold(f"  [{task_name}]"))
        t0 = time.perf_counter()

        benchmark_cls = benchmark_classes[task_name]
        benchmark = benchmark_cls(code_language=language, provider=provider)

        stats, raw = test_task(task_name, benchmark, mapper, language, target_language, show_dims=show_dims)
        stats["elapsed_s"] = round(time.perf_counter() - t0, 2)

        if stats["skipped"]:
            print(f"      {yellow('SKIPPED')}  {stats['skip_reason']}")
        elif stats["records_error"] == stats["records_attempted"] and stats["records_attempted"] > 0:
            print(f"      {red('ALL RECORDS FAILED')}")

        all_stats.append(stats)
        all_raw_results.extend(raw)

    wall_elapsed = time.perf_counter() - wall_start

    # -----------------------------------------------------------------------
    # Score via ScoringEngine (same as TestSuite.get_summary)
    # -----------------------------------------------------------------------
    from core.scoring import ScoringEngine

    scoring_report = ScoringEngine(all_raw_results).compute()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print(bold(cyan("─" * 72)))
    print(bold(cyan("  Summary")))
    print(bold(cyan("─" * 72)))
    print()
    print(f"  {'Task':<22}  {'Status':<8}  {'OK/Attempted':>12}  {'Errors':>6}  {'Mean Score':>10}  {'Time':>6}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*12}  {'-'*6}  {'-'*10}  {'-'*6}")

    total_ok = 0
    total_attempted = 0
    total_errors = 0
    task_pass_count = 0

    for s in all_stats:
        if s["skipped"]:
            status    = yellow("SKIPPED")
            ok_str    = dim("N/A")
            err_str   = dim("N/A")
            score_str = dim("N/A")
            time_str  = dim("N/A")
        else:
            all_failed = s["records_error"] == s["records_attempted"] and s["records_attempted"] > 0
            status    = red("FAIL") if all_failed else green("PASS")
            if not all_failed:
                task_pass_count += 1
            ok_str    = f"{s['records_ok']}/{s['records_attempted']}"
            err_str   = str(s["records_error"]) if s["records_error"] else dim("0")
            ms        = scoring_report.task_scores.get(s["task"])
            score_str = f"{ms.mean_score:.4f}" if ms and ms.scored_count > 0 else dim("N/A")
            time_str  = f"{s['elapsed_s']}s"
            total_ok        += s["records_ok"]
            total_attempted += s["records_attempted"]
            total_errors    += s["records_error"]

        print(f"  {s['task']:<22}  {status:<8}  {ok_str:>12}  {err_str:>6}  {score_str:>10}  {time_str:>6}")

    print()
    print(f"  Final score : {bold(f'{scoring_report.final_score:.4f}')}  |  Grade: {bold(scoring_report.grade)}")
    print(f"  Total: {total_ok}/{total_attempted} records OK  |  {total_errors} errors  |  "
          f"{task_pass_count}/{len(tasks_to_run)} tasks passed  |  {wall_elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Print errors in detail
    # -----------------------------------------------------------------------
    any_errors = any(s.get("errors") for s in all_stats)
    if any_errors:
        print()
        print(bold(red("  Errors:")))
        for s in all_stats:
            for err in s.get("errors", []):
                print(f"    [{s['task']}] record {err['record_id']}: {err['error']}")

    # -----------------------------------------------------------------------
    # Save report — same structure as run_experiment.py
    # -----------------------------------------------------------------------
    out_dir = Path("reports/outputs") / language
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"{ts}_smoke_test_{model_name.replace('/', '-')}.json"

    summary = {
        "suite_name": f"smoke_test_{model_name}",
        "elapsed_s": round(wall_elapsed, 2),
        "final_score": scoring_report.final_score,
        "grade": scoring_report.grade,
        "overall_pass_rate": scoring_report.overall_pass_rate,
        "category_scores": scoring_report.category_scores,
        "task_scores": {k: v.to_dict() for k, v in scoring_report.task_scores.items()},
        "total": scoring_report.total_records,
        "passed": scoring_report.total_passed,
        "failed": scoring_report.total_records - scoring_report.total_passed - scoring_report.total_errors,
        "errors": scoring_report.total_errors,
        "pass_rate": scoring_report.overall_pass_rate,
    }

    report = {
        "metadata": {
            "run_name": f"smoke_test_{model_name}",
            "timestamp": ts,
            "language": language,
            "limit_per_task": LIMIT,
            "tasks_run": tasks_to_run,
        },
        "model_config": {
            "provider":     provider_name,
            "model_name":   model_name,
            "base_url":     base_url,
            "temperature":  temperature,
            "max_tokens":   max_tokens,
            "system_prompt": system_prompt,
        },
        "summary": summary,
        "results": [r.to_dict() for r in all_raw_results],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print()
    print(f"  Report saved: {green(str(report_path))}")
    print()

    # Exit 1 if any task fully failed (not just skipped)
    fully_failed = [s for s in all_stats if not s["skipped"] and s["records_error"] == s["records_attempted"] > 0]
    sys.exit(1 if fully_failed else 0)


if __name__ == "__main__":
    main()

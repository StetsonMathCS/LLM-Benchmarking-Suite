"""Publication-quality figures rendered from saved FACETS results.

Every figure is produced from already-saved run data: nothing here calls a
provider, and no figure mixes incompatible cohorts. Each figure is written as
PNG (raster preview), PDF (print), and SVG (editable vector), with a caption and
the exact table it was drawn from recorded alongside it.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median

FIGURE_FORMATS = ("png", "pdf", "svg")
RASTER_DPI = 300

# Colorblind-safe qualitative sequence; reused by every figure so a model keeps
# one color across the whole figure set.
MODEL_COLORS = (
    "#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951",
    "#ff8ab7", "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0",
)

MODEL_SIZES = {
    "phi4-mini:3.8b": 3.8,
    "ministral-3:3b": 3.0,
    "mistral:7b": 7.0,
    "gpt-oss:20b": 20.0,
    "deepseek-r1:32b": 32.0,
    "llama3:70b": 70.0,
    "qwen3.6:35b": 35.0,
    "qwen3.8": 27.0,
    "nemotron-3.5-lightning": 30.0,
}

TASK_LABELS = {
    "bug_fixing": "Bug Fixing",
    "code_generation": "Code Generation",
    "code_review": "Code Review",
    "refactoring": "Refactoring",
    "test_generation": "Test Generation",
    "translation": "Translation",
}

PROVIDER_COLORS = {
    "openai": "#4C72B0",
    "anthropic": "#C44E52",
    "ollama": "#2ca02c",
}

STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": RASTER_DPI,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "legend.frameon": False,
    "figure.autolayout": False,
}


def color_for(index: int) -> str:
    return MODEL_COLORS[index % len(MODEL_COLORS)]


def palette(models: Sequence[str]) -> dict[str, str]:
    return {model: color_for(index) for index, model in enumerate(models)}


def _save(fig, output: Path, stem: str, caption: str, **metadata) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for extension in FIGURE_FORMATS:
        path = output / f"{stem}.{extension}"
        if extension == "png":
            fig.savefig(path, dpi=RASTER_DPI)
        else:
            fig.savefig(path)
        files[extension] = path.name
    import matplotlib.pyplot as plt

    plt.close(fig)
    return {"stem": stem, "caption": caption, "files": files, **metadata}


def _nice_limits(values: Sequence[float], pad: float = 0.04, min_span: float = 0.10) -> tuple[float, float]:
    """Padded axis limits, but never so zoomed that trivial gaps look decisive."""
    low, high = min(values), max(values)
    if math.isclose(low, high):
        center = low
        low, high = center - min_span / 2, center + min_span / 2
    span = high - low
    if span < min_span:
        center = (low + high) / 2
        low, high = center - min_span / 2, center + min_span / 2
    else:
        low, high = low - pad * span, high + pad * span
    return low, high


def violin_by_model(records: list[dict], output: Path, models: Sequence[str], tasks: Sequence[str]) -> dict | None:
    """Record-level score distributions, one panel per task, one violin per model."""
    if not records or not models:
        return None
    import matplotlib.pyplot as plt

    tasks = [task for task in tasks if any(
        row["model"] in models and row["task"] == task and row["combined_score"] is not None
        for row in records
    )]
    if not tasks:
        return None
    colors = palette(models)
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(2.6 * len(tasks), 3.4), squeeze=False, sharey=True
    )
    for axis, task in zip(axes[0], tasks, strict=True):
        present = [model for model in models if any(
            row["model"] == model and row["task"] == task and row["combined_score"] is not None
            for row in records
        )]
        result = axis.violinplot(
            [
                [row["combined_score"] for row in records
                 if row["model"] == model and row["task"] == task and row["combined_score"] is not None]
                for model in present
            ],
            showmeans=True, showextrema=False, widths=0.8,
        )
        for body, model in zip(result["bodies"], present, strict=True):
            body.set_facecolor(colors[model])
            body.set_alpha(0.55)
            body.set_edgecolor(colors[model])
        axis.set_xticks(range(1, len(present) + 1))
        axis.set_xticklabels(present, rotation=60, ha="right", fontsize=7)
        axis.set_title(task.replace("_", " "), fontsize=9)
        axis.set_ylim(0, 1)
    axes[0][0].set_ylabel("record composite score")
    return _save(
        fig, output, "record_score_distributions",
        "Record-level composite score distribution per model, one panel per task. "
        "Width shows where records concentrate; the line marks the mean. Records are the "
        "unit of variation, so spread reflects record difficulty and model behaviour, not "
        "repeated-inference uncertainty.",
        models=list(models), tasks=list(tasks),
    )


def radar_tasks(task_summary: list[dict], output: Path, models: Sequence[str], tasks: Sequence[str]) -> dict | None:
    """Six-task capability profile per model on a shared 0-1 axis."""
    if not task_summary or len(tasks) < 3:
        return None
    import matplotlib.pyplot as plt

    angles = [index / len(tasks) * 2 * math.pi for index in range(len(tasks))]
    angles.append(angles[0])
    colors = palette(models)
    fig, axis = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"polar": True})
    for model in models:
        scores = []
        for task in tasks:
            row = next(
                (entry for entry in task_summary if entry["model"] == model and entry["task"] == task),
                None,
            )
            scores.append(float(row["mean_score"]) if row and row["mean_score"] is not None else 0.0)
        closed = scores + [scores[0]]
        axis.plot(angles, closed, color=colors[model], linewidth=1.6, label=model)
        axis.fill(angles, closed, color=colors[model], alpha=0.10)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels([task.replace("_", " ") for task in tasks], fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_yticks([0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    axis.legend(loc="upper right", bbox_to_anchor=(1.22, 1.1), fontsize=8)
    return _save(
        fig, output, "task_capability_radar",
        "Mean composite score per task for each model. Axes run 0-1 and are ordered "
        "identically for every model, so polygon area compares breadth across tasks but "
        "must not be read as a weighted composite.",
        models=list(models), tasks=list(tasks),
    )


def heatmap_model_task(task_summary: list[dict], output: Path, models: Sequence[str], tasks: Sequence[str]) -> dict | None:
    """Model-by-task mean score grid for quick pattern reading."""
    if not task_summary:
        return None
    import matplotlib.pyplot as plt

    lookup = {
        (row["model"], row["task"]): row["mean_score"]
        for row in task_summary
        if row["mean_score"] is not None
    }
    if not lookup:
        return None
    grid = [[lookup.get((model, task)) for task in tasks] for model in models]
    fig, axis = plt.subplots(figsize=(1.5 * len(tasks) + 2.5, 0.6 * len(models) + 2))
    image = axis.imshow(
        [[float("nan") if value is None else value for value in row] for row in grid],
        cmap="viridis", vmin=0, vmax=1, aspect="auto",
    )
    axis.set_xticks(range(len(tasks)))
    axis.set_xticklabels([task.replace("_", " ") for task in tasks], rotation=40, ha="right")
    axis.set_yticks(range(len(models)))
    axis.set_yticklabels(models)
    axis.grid(False)
    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            if value is None:
                axis.text(column_index, row_index, "n/a", ha="center", va="center", fontsize=7, color="grey")
            else:
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7,
                          color="white" if value < 0.55 else "black")
    bar = fig.colorbar(image, ax=axis, shrink=0.85)
    bar.set_label("mean record composite score")
    return _save(
        fig, output, "model_task_score_heatmap",
        "Mean composite score for every model-task pair. Cells marked n/a had no scored "
        "records, which is a coverage gap rather than a zero.",
        models=list(models), tasks=list(tasks),
    )


def grouped_task_bars(task_summary: list[dict], output: Path, models: Sequence[str], tasks: Sequence[str]) -> dict | None:
    """Task-by-task comparison with models side by side."""
    if not task_summary or not models:
        return None
    import matplotlib.pyplot as plt
    import numpy as np

    colors = palette(models)
    fig, axis = plt.subplots(figsize=(max(7, 1.6 * len(tasks)), 3.8))
    positions = np.arange(len(tasks))
    width = 0.8 / max(1, len(models))
    for index, model in enumerate(models):
        heights = []
        for task in tasks:
            row = next((entry for entry in task_summary if entry["model"] == model and entry["task"] == task), None)
            heights.append(float(row["mean_score"]) if row and row["mean_score"] is not None else 0.0)
        axis.bar(positions + index * width, heights, width=width, label=model, color=colors[model])
    axis.set_xticks(positions + 0.4 - width / 2)
    axis.set_xticklabels([task.replace("_", " ") for task in tasks], rotation=25, ha="right")
    axis.set_ylabel("mean record composite score")
    axis.set_ylim(0, 1)
    axis.legend(ncols=min(3, len(models)), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    return _save(
        fig, output, "task_score_grouped_bars",
        "Mean record composite score per task with models grouped side by side. Missing "
        "combinations are drawn as zero height and are listed as n/a in task_summary.csv.",
        models=list(models), tasks=list(tasks),
    )


def score_vs_parameters(
    model_scores: dict[str, float], parameter_sizes: dict[str, float], output: Path,
    models: Sequence[str], undisclosed: Sequence[str],
) -> dict | None:
    """Composite score against published parameter size, known sizes only."""
    points = [
        (model, parameter_sizes[model], model_scores[model])
        for model in models
        if model in parameter_sizes and model_scores.get(model) is not None and parameter_sizes[model] > 0
    ]
    if len(points) < 2:
        return None
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.set_xscale("log")
    for model, size, score in points:
        axis.scatter(size, score, s=70, color=color_for(list(model_scores).index(model)), zorder=3)
        axis.annotate(model, (size, score), textcoords="offset points", xytext=(6, 4), fontsize=8)
    axis.set_xlabel("published parameter size (billions, log scale)")
    axis.set_ylabel("FACETS composite score")
    axis.set_ylim(*_nice_limits([score for _, _, score in points]))
    if undisclosed:
        axis.annotate(
            f"excluded (size undisclosed/unverified): {', '.join(undisclosed)}",
            xy=(0.01, 0.02), xycoords="axes fraction", fontsize=7, color="grey",
        )
    return _save(
        fig, output, "score_vs_parameter_size",
        "Composite score against published parameter size. Models whose size is "
        "undisclosed or unverified are excluded rather than assumed, so the axis is not a "
        "complete survey of the cohort and no scaling law is claimed.",
        models=[model for model, _, _ in points], excluded=list(undisclosed),
    )


def score_vs_latency(
    latency_rows: list[dict], model_scores: dict[str, float], output: Path, models: Sequence[str],
) -> dict | None:
    """Composite score against measured generation latency, per model median."""
    per_model: dict[str, list[float]] = {}
    for row in latency_rows:
        if row.get("latency_ms") is not None and row["model"] in model_scores:
            per_model.setdefault(row["model"], []).append(float(row["latency_ms"]))
    points = [
        (model, median(values), model_scores[model])
        for model, values in per_model.items()
        if values and model_scores.get(model) is not None and median(values) > 0
    ]
    if len(points) < 2:
        return None
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.set_xscale("log")
    for model, latency, score in sorted(points):
        axis.scatter(latency, score, s=70, color=color_for(list(model_scores).index(model)), zorder=3)
        axis.annotate(model, (latency, score), textcoords="offset points", xytext=(6, 4), fontsize=8)
    axis.set_xlabel("median provider-reported generation latency (ms, log scale)")
    axis.set_ylabel("FACETS composite score")
    axis.set_ylim(*_nice_limits([score for _, _, score in points]))
    return _save(
        fig, output, "score_vs_generation_latency",
        "Composite score against median provider-reported generation latency per model. "
        "Latency reflects the recorded hardware and provider path of these runs only and is "
        "not a controlled cross-provider benchmark.",
        models=[model for model, _, _ in sorted(points)],
    )


def _provider_for_model(model: str) -> str:
    if model.startswith("gpt") or model.startswith("o3") or model.startswith("o4") or model in {"gpt-5-codex", "gpt-5.4", "gpt-5-nano"}:
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    return "ollama"


def _color_for_model(model: str, models: Sequence[str]) -> str:
    provider = _provider_for_model(model)
    if provider == "openai":
        return "#4C72B0"
    if provider == "anthropic":
        return "#C44E52"
    if provider == "ollama":
        return "#2ca02c"
    return color_for(list(models).index(model))


def _annotate_no_overlap(ax, points, fontsize=8):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    offsets = [(6, 4), (6, -12), (6, 20), (6, -28), (6, 36), (-70, 4), (6, -44)]
    boxes = []
    for x, y, label, color in points:
        placed = None
        for dx, dy in offsets:
            text = ax.annotate(label, (x, y), textcoords="offset points",
                               xytext=(dx, dy), fontsize=fontsize, color=color)
            bb = text.get_window_extent(renderer=renderer)
            if not any(bb.overlaps(b) for b in boxes):
                boxes.append(bb)
                placed = text
                break
            text.remove()
        if placed is None:
            placed = ax.annotate(label, (x, y), textcoords="offset points",
                                 xytext=(6, 4), fontsize=fontsize, color=color)
            boxes.append(placed.get_window_extent(renderer=renderer))


def cmp_score_vs_size(models: list[str], model_scores: dict[str, float], output: Path) -> dict | None:
    local = [m for m in models if m in MODEL_SIZES]
    if not local:
        return None
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5.5))
    points = []
    for m in local:
        size = MODEL_SIZES[m]
        score = model_scores.get(m) * 100 if model_scores.get(m) is not None else None
        if score is None:
            continue
        c = _color_for_model(m, models)
        ax.scatter(size, score, s=140, color=c, zorder=5, edgecolors="white", linewidths=1.2)
        points.append((size, score, m, c))
    if len(points) >= 2:
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        z = np.polyfit(xs, ys, 1)
        px = np.linspace(min(xs) - 1, max(xs) + 2, 100)
        ax.plot(px, np.polyval(z, px), "--", color="#888", linewidth=1.2, label="Trend (local)", zorder=3)
    _annotate_no_overlap(ax, points)
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("Final Score (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.xaxis.set_major_locator(mticker.FixedLocator([0, 20, 40, 60]))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.set_xlim(0, 76)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, linestyle=":")
    return _save(fig, output, "cmp_score_vs_size",
                 "Model performance versus published parameter count (local models only).")


def cmp_task_difficulty(task_summary: list[dict], output: Path, models: list[str], tasks: list[str]) -> dict | None:
    if not task_summary or not models:
        return None
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [t for t in tasks if t in TASK_LABELS]
    if not tasks:
        return None
    means, stds = [], []
    for task in tasks:
        vals = [row["mean_score"] for row in task_summary
                if row["task"] == task and row["mean_score"] is not None]
        if not vals:
            return None
        means.append(np.mean(vals) * 100)
        stds.append(np.std(vals) * 100)
    cmap = plt.cm.RdYlGn
    colors = [cmap(v / 100) for v in means]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(tasks))
    bars = ax.bar(x, means, yerr=stds, color=colors, zorder=3, edgecolor="white", capsize=6,
                  error_kw=dict(color="#333", linewidth=1.4, capthick=1.4))
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 2,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=10, rotation=20, ha="right")
    ax.set_ylim(0, 116)
    ax.axhline(80, color="#888", linewidth=1, linestyle=":", zorder=2, label="80% threshold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    return _save(fig, output, "cmp_task_difficulty",
                 "Mean task performance across all models with standard deviation.")


def cmp_score_violin(records: list[dict], output: Path, models: list[str]) -> dict | None:
    if not records or not models:
        return None
    import matplotlib.pyplot as plt
    import numpy as np

    all_scores = []
    colors = []
    for m in models:
        scores = [row["combined_score"] for row in records
                  if row["model"] == m and row["combined_score"] is not None]
        if not scores:
            continue
        all_scores.append(scores)
        colors.append(_color_for_model(m, models))
    if not all_scores:
        return None
    fig, ax = plt.subplots(figsize=(13, 6))
    parts = ax.violinplot(all_scores, positions=range(len(all_scores)), showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.45)
        pc.set_edgecolor(color)
    for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
        parts[part].set_color("#333")
        parts[part].set_linewidth(1.0)
    rng = np.random.default_rng(42)
    for i, (scores, color) in enumerate(zip(all_scores, colors)):
        jitter = rng.uniform(-0.14, 0.14, size=len(scores))
        ax.scatter(i + jitter, scores, color=color, alpha=0.45, s=10, zorder=3, edgecolors="none")
    ax.set_xticks(range(len(all_scores)))
    ax.set_xticklabels([models[i] for i in range(len(models)) if any(r["model"] == models[i] and r["combined_score"] is not None for r in records)], fontsize=8, rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.8, color="#888", linewidth=1, linestyle=":", zorder=2, label="0.80 threshold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    return _save(fig, output, "cmp_score_violin",
                 "Distribution of individual task scores across all models.")


def cmp_radar_all(task_summary: list[dict], output: Path, models: list[str], tasks: list[str]) -> dict | None:
    if not task_summary or not models:
        return None
    tasks = [t for t in tasks if t in TASK_LABELS]
    if len(tasks) < 3:
        return None
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [TASK_LABELS[t] for t in tasks]
    N = len(tasks)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_plot = angles + [angles[0]]
    line_styles = ["-", ":"]
    markers = ["o", "s", "^", "D", "v", "p", "h", "X", "*"]

    cloud = [m for m in models if _provider_for_model(m) in ("openai", "anthropic")]
    local = [m for m in models if _provider_for_model(m) == "ollama"]

    def _draw_panel(ax, group):
        for r, lbl in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%"), (1.0, "100%")]:
            ax.plot(angles_plot, [r] * (N + 1), color="#ccc", linewidth=0.9, linestyle="--", zorder=1)
            ax.text(angles[0], r + 0.03, lbl, ha="center", fontsize=8, color="#AAAAAA")
        handles = []
        for i, m in enumerate(group):
            values = []
            for t in tasks:
                row = next((e for e in task_summary if e["model"] == m and e["task"] == t), None)
                values.append(float(row["mean_score"]) if row and row["mean_score"] is not None else 0.0)
            values_plot = values + [values[0]]
            c = _color_for_model(m, models)
            ls = line_styles[i % len(line_styles)]
            mk = markers[i % len(markers)]
            line, = ax.plot(angles_plot, values_plot, color=c, linewidth=2.0, linestyle=ls,
                            marker=mk, markersize=6, markerfacecolor=c, markeredgecolor="white",
                            label=m, zorder=3)
            handles.append(line)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
        ax.set_yticks([])
        ax.set_ylim(0, 1.08)
        return handles

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5), subplot_kw=dict(polar=True))
    handles = _draw_panel(ax1, cloud) + _draw_panel(ax2, local)
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="lower center", ncol=4, fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.98, bottom=0.20, wspace=0.30)
    return _save(fig, output, "cmp_radar_all",
                 "Radar comparison across tasks with cloud and local models separated.")


RENDERERS: tuple[Callable[..., dict | None], ...] = (
    violin_by_model,
    heatmap_model_task,
    grouped_task_bars,
    score_vs_parameters,
    score_vs_latency,
    cmp_score_vs_size,
    cmp_task_difficulty,
    cmp_score_violin,
    cmp_radar_all,
)

RENDERABLE_FAILURES = (AttributeError, IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError)


def render_figures(
    *,
    output: Path,
    records: list[dict],
    task_summary: list[dict],
    model_scores: dict[str, float],
    parameter_sizes: dict[str, float],
    undisclosed_sizes: Sequence[str],
    latency_rows: list[dict],
    models: Sequence[str],
    tasks: Sequence[str],
) -> dict:
    """Render every figure that has enough data and describe the result.

    A figure that cannot be drawn is recorded as a failure rather than aborting the
    analysis, so one empty panel cannot cost the whole figure set.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_task = {"output": output, "models": list(models), "tasks": list(tasks)}
    payloads = {
        violin_by_model: {**per_task, "records": records},
        radar_tasks: {**per_task, "task_summary": task_summary},
        heatmap_model_task: {**per_task, "task_summary": task_summary},
        grouped_task_bars: {**per_task, "task_summary": task_summary},
        score_vs_parameters: {
            "output": output, "models": list(models),
            "model_scores": model_scores, "parameter_sizes": parameter_sizes,
            "undisclosed": list(undisclosed_sizes),
        },
        score_vs_latency: {
            "output": output, "models": list(models),
            "latency_rows": latency_rows, "model_scores": model_scores,
        },
        cmp_score_vs_size: {"models": list(models), "model_scores": model_scores, "output": output},
        cmp_task_difficulty: {"task_summary": task_summary, "output": output, "models": list(models), "tasks": list(tasks)},
        cmp_score_violin: {"records": records, "output": output, "models": list(models)},
        cmp_radar_all: {"task_summary": task_summary, "output": output, "models": list(models), "tasks": list(tasks)},
    }
    figures = []
    failures = []
    with plt.rc_context(STYLE):
        for renderer in RENDERERS:
            try:
                entry = renderer(**payloads[renderer])
            except RENDERABLE_FAILURES as exc:
                failures.append(f"{renderer.__name__}: {exc}")
                continue
            if entry:
                figures.append(entry)
    return {
        "figures": figures,
        "stems": [entry["stem"] for entry in figures],
        "formats": list(FIGURE_FORMATS),
        "failures": failures,
        "note": "figures are rendered from saved records only; no provider call is made",
    }


def write_index(output: Path, figures: list[dict], context: dict) -> Path:
    """Markdown index with captions and the provenance of every figure."""
    lines = [
        "# FACETS figures",
        "",
        (
            f"Generated {context.get('generated_at')} from {context.get('runs')} run(s) under evaluator "
            f"`{context.get('evaluator_version')}` and profile `{context.get('scoring_profile')}`."
        ),
        "",
        f"Models: {', '.join(context.get('models', [])) or 'none'}",
        "",
    ]
    if context.get("warnings"):
        lines += ["## Warnings", ""] + [f"- {warning}" for warning in context["warnings"]] + [""]
    lines += ["## Figures", ""]
    for entry in figures:
        png = entry["files"].get("png", "")
        lines += [
            f"### {entry['stem']}",
            "",
            f"![{entry['stem']}]({png})",
            "",
            entry["caption"],
            "",
            f"Vector copies: {', '.join(entry['files'].get(fmt, '') for fmt in ('pdf', 'svg'))}",
            "",
        ]
    index = output / "index.md"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def figure_manifest(result: dict, output: Path, index: Path | None) -> dict:
    payload = {**result, "index": index.name if index else None}
    (output / "figures.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

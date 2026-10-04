"""Publication-quality figures rendered from saved FACETS results.

Every figure is produced from already-saved run data: nothing here calls a
provider, and no figure mixes incompatible cohorts. Each figure is written as
PNG (raster preview), PDF (print), and SVG (editable vector), with a caption and
the exact table it was drawn from recorded alongside it.
"""

from __future__ import annotations

import json
import math
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


RENDERERS: tuple[Callable[..., dict | None], ...] = (
    violin_by_model,
    radar_tasks,
    heatmap_model_task,
    grouped_task_bars,
    score_vs_parameters,
    score_vs_latency,
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

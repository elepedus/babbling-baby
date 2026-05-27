"""Plot the trace metrics from a training run's trace.jsonl.

Includes the four-granularity metrics, the curriculum focus level
over time, and per-level success rates.

Usage:
    uv run python scripts/plot_trace.py runs/<run_id>/trace.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("percent_printable", "% printable ASCII"),
    ("percent_letters", "% letters [a-zA-Z]"),
    ("percent_shape", "% plausible word-shape (runs 3–12)"),
    ("percent_dict", "% dictionary words"),
]


def plot_trace(jsonl_path: Path, out_path: Path | None = None) -> Path:
    points = [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    if not points:
        raise ValueError(f"No trace points in {jsonl_path}")

    elapsed = [p["elapsed_s"] for p in points]
    has_level = any("level" in p for p in points)
    has_per_level = any("per_level_rate" in p for p in points)

    if has_per_level:
        fig = plt.figure(figsize=(14, 13))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.6, 0.9])
        metric_axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]
        ax_level = fig.add_subplot(gs[2, :])
        ax_per_level = fig.add_subplot(gs[3, :])
    elif has_level:
        fig = plt.figure(figsize=(13, 11))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])
        metric_axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]
        ax_level = fig.add_subplot(gs[2, :])
        ax_per_level = None
    else:
        fig, axes2d = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        metric_axes = list(axes2d.flat)
        ax_level = None
        ax_per_level = None

    transitions: list[tuple[float, int]] = []
    if has_level:
        prev = None
        for p in points:
            lvl = p.get("level")
            if lvl is not None and lvl != prev:
                transitions.append((p["elapsed_s"], lvl))
                prev = lvl

    for (key, label), ax in zip(METRICS, metric_axes):
        ax.plot(elapsed, [p[key] for p in points])
        ax.set_xlabel("training time (s)")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(label)
        for t, _lvl in transitions[1:]:
            ax.axvline(t, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)

    if ax_level is not None:
        levels = [p.get("level", 0) for p in points]
        ax_level.step(elapsed, levels, where="post")
        ax_level.set_xlabel("training time (s)")
        ax_level.set_ylabel("focus level")
        ax_level.set_ylim(-0.5, 5.5)
        ax_level.set_yticks(range(6))
        level_names_seen: dict[int, str] = {}
        for p in points:
            lvl = p.get("level")
            name = p.get("level_name")
            if lvl is not None and name and lvl not in level_names_seen:
                level_names_seen[lvl] = name
        if level_names_seen:
            tick_labels = [
                level_names_seen.get(i, str(i)) for i in range(6)
            ]
            ax_level.set_yticklabels(tick_labels)
        ax_level.grid(True, alpha=0.3)
        ax_level.set_title("curriculum focus level")

    if ax_per_level is not None:
        per_level_lists: list[list[float]] = [[] for _ in range(6)]
        for p in points:
            row = p.get("per_level_rate") or []
            for i, v in enumerate(row[:6]):
                per_level_lists[i].append(v)
        cmap = plt.cm.viridis
        for L, vals in enumerate(per_level_lists):
            if not vals or all(v == 0 for v in vals):
                continue
            ax_per_level.plot(
                elapsed[: len(vals)],
                vals,
                label=f"level {L}",
                color=cmap(L / 5),
            )
        ax_per_level.set_xlabel("training time (s)")
        ax_per_level.set_ylabel("per-level success rate")
        ax_per_level.set_ylim(0, 1.05)
        ax_per_level.grid(True, alpha=0.3)
        ax_per_level.set_title("per-level rolling success rate (hit at exactly this level)")
        ax_per_level.legend(loc="upper left", fontsize=9)
        # Overlay each level's unlock threshold as a horizontal line
        from babbling_baby.curriculum import Curriculum
        for L, thresh in enumerate(Curriculum.UNLOCK_THRESHOLDS):
            ax_per_level.axhline(
                thresh, color=cmap(L / 5), linestyle=":", alpha=0.5, linewidth=0.8
            )

    fig.suptitle(f"Babbling Baby — trace from {jsonl_path.parent.name}", fontsize=14)
    fig.tight_layout()

    if out_path is None:
        out_path = jsonl_path.parent / "trace.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_trace.py <path/to/trace.jsonl>", file=sys.stderr)
        sys.exit(1)
    out = plot_trace(Path(sys.argv[1]))
    print(f"wrote {out}")

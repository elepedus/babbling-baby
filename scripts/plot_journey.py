"""Plot the journey across all training runs.

Produces a single comparison figure that overlays %dict, %letters,
and (where present) the focus-level progression across the
experimental runs. The 8-hour overnight run is plotted on a separate
axis so its scale doesn't compress the 10-minute runs.

Usage:
    uv run python scripts/plot_journey.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RUNS_DIR = Path("runs")


# Ordered list of runs to compare. Each entry: (run_id, label, colour, kind).
# kind is "loose" (loose verifier) or "strict" (strict verifier).
RUNS = [
    ("20260526-192431", "loose verifier (no delim)",  "tab:gray",   "loose"),
    ("20260526-195231", "strict verifier, no delim",  "tab:olive",  "strict"),
    ("20260526-200538", "strict + space delim",       "tab:brown",  "strict"),
    ("20260526-203202", "+ flat-reward RL",           "tab:red",    "strict"),
    ("20260526-210157", "+ length-aware RL",          "tab:purple", "strict"),
    ("20260526-212829", "+ discrete-jump curric.",    "tab:orange", "strict"),
    ("20260526-215121", "+ graded curriculum",        "tab:blue",   "strict"),
]
OVERNIGHT_RUN = ("20260526-220809", "overnight (8h, graded)", "tab:green", "strict")


def load_trace(run_id: str) -> list[dict]:
    path = RUNS_DIR / run_id / "trace.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_levels(run_id: str) -> list[dict]:
    path = RUNS_DIR / run_id / "levels.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    fig = plt.figure(figsize=(13, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.2], hspace=0.45)

    ax_letters = fig.add_subplot(gs[0])
    ax_dict_short = fig.add_subplot(gs[1])
    ax_dict_long = fig.add_subplot(gs[2])
    ax_level = fig.add_subplot(gs[3])

    # --- Top: %letters across the seven 10-min runs ---
    for run_id, label, colour, _kind in RUNS:
        points = load_trace(run_id)
        if not points:
            continue
        xs = [p["elapsed_s"] for p in points]
        ys = [p["percent_letters"] for p in points]
        ax_letters.plot(xs, ys, label=label, color=colour, linewidth=1.4, alpha=0.85)
    ax_letters.set_title("%letters across the 10-minute design iterations", fontsize=12)
    ax_letters.set_xlabel("training time (s)")
    ax_letters.set_ylabel("% letters [a-zA-Z]")
    ax_letters.set_ylim(0, 1.05)
    ax_letters.grid(True, alpha=0.3)
    ax_letters.legend(loc="lower left", fontsize=8, ncol=2)

    # --- Middle: %dict across the 10-min runs ---
    for run_id, label, colour, kind in RUNS:
        points = load_trace(run_id)
        if not points:
            continue
        xs = [p["elapsed_s"] for p in points]
        ys = [p["percent_dict"] for p in points]
        style = "--" if kind == "loose" else "-"
        ax_dict_short.plot(
            xs, ys, label=label, color=colour, linewidth=1.4,
            linestyle=style, alpha=0.85,
        )
    ax_dict_short.set_title(
        "%dict across the 10-minute design iterations  "
        "(dashed = loose verifier, solid = strict)",
        fontsize=12,
    )
    ax_dict_short.set_xlabel("training time (s)")
    ax_dict_short.set_ylabel("% dictionary words")
    ax_dict_short.set_ylim(0, 0.55)
    ax_dict_short.grid(True, alpha=0.3)
    ax_dict_short.legend(loc="upper right", fontsize=8, ncol=2)

    # --- Lower middle: the overnight %dict trajectory ---
    points = load_trace(OVERNIGHT_RUN[0])
    if points:
        xs = [p["elapsed_s"] / 3600 for p in points]
        ys = [p["percent_dict"] for p in points]
        ax_dict_long.plot(xs, ys, color=OVERNIGHT_RUN[2], linewidth=1.5)
        ax_dict_long.set_title(
            "%dict across the 8-hour overnight run (same architecture, same curriculum)",
            fontsize=12,
        )
        ax_dict_long.set_xlabel("training time (hours)")
        ax_dict_long.set_ylabel("% dictionary words")
        ax_dict_long.set_ylim(0, 0.35)
        ax_dict_long.grid(True, alpha=0.3)

    # --- Bottom: focus level for the curriculum-driven runs ---
    discrete_run = "20260526-212829"
    graded_run = "20260526-215121"
    overnight_run = OVERNIGHT_RUN[0]

    for run_id, label, colour, in (
        (discrete_run, "discrete-jump, 10 min", "tab:orange"),
        (graded_run, "graded, 10 min", "tab:blue"),
        (overnight_run, "graded, 8 hours", "tab:green"),
    ):
        points = load_trace(run_id)
        if not points:
            continue
        # Use hours if it's the overnight run, seconds otherwise
        is_long = run_id == overnight_run
        if is_long:
            xs = [p["elapsed_s"] / 3600 for p in points]
        else:
            xs = [p["elapsed_s"] / 60 for p in points]
        ys = [p.get("level", 0) for p in points]
        ax_level.step(
            xs, ys, where="post", label=label, color=colour, linewidth=1.6,
        )
    ax_level.set_title(
        "Focus level over time  (10-min axis in minutes; 8-h axis in hours)",
        fontsize=12,
    )
    ax_level.set_xlabel("training time (minutes for short runs / hours for overnight)")
    ax_level.set_ylabel("curriculum focus level")
    ax_level.set_ylim(-0.3, 5.3)
    ax_level.set_yticks(range(6))
    ax_level.set_yticklabels([
        "L0\n1-letter", "L1\n2-letter", "L2\n3-letter",
        "L3\n4-5", "L4\n6-8", "L5\nfull",
    ])
    ax_level.grid(True, alpha=0.3)
    ax_level.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Babbling Baby — design iteration trajectories",
        fontsize=14, y=0.995,
    )
    fig.savefig("runs/journey.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote runs/journey.png")


if __name__ == "__main__":
    main()

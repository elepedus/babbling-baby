"""Run a longer training session with trace logging.

Usage:
    uv run python scripts/run_training.py [duration_seconds] [log_interval_s]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from babbling_baby.train import train


def main(duration_s: int = 600, log_interval_s: float = 5.0) -> None:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / run_id
    print(f"run dir: {run_dir}")
    print(f"duration: {duration_s}s")
    print(f"log interval: {log_interval_s}s")

    start = time.time()
    pupil = train(
        time_budget_seconds=duration_s,
        log_interval_s=log_interval_s,
        run_dir=run_dir,
        sample_n_bytes=1024,
    )
    elapsed = time.time() - start
    print(f"training finished in {elapsed:.1f}s")

    weights_path = run_dir / "pupil.npz"
    pupil.save_weights(str(weights_path))
    print(f"saved weights to {weights_path}")

    trace_path = run_dir / "trace.jsonl"
    lines = trace_path.read_text().splitlines()
    print(f"\nlogged {len(lines)} trace points")

    print("\n--- first and last trace points ---")
    if lines:
        first = json.loads(lines[0])
        last = json.loads(lines[-1])
        for label, p in (("first", first), ("last", last)):
            print(f"{label}:")
            for k in ("elapsed_s", "loss_ema", "level", "level_name",
                     "level_success_rate", "percent_letters",
                     "percent_shape", "percent_dict"):
                v = p.get(k)
                if v is not None:
                    print(f"  {k:20s} {v}")
            preview = p.get("sample_preview", "")
            if preview:
                print(f"  sample: {preview[:96]!r}")

    levels_path = run_dir / "levels.jsonl"
    if levels_path.exists():
        print("\n--- focus transitions ---")
        for line in levels_path.read_text().splitlines():
            if line.strip():
                ev = json.loads(line)
                print(f"  t={ev['elapsed_s']:7.1f}s  -> focus {ev['level']}")


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    log_interval = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    main(duration, log_interval)

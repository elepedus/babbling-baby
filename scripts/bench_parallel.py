"""Benchmark how training step time scales with N parallel pupils.

If the GPU is compute-unsaturated on our small model, we should be
able to train N pupils for roughly the same wall-clock as one. The
benchmark times 100 training steps for each N in {1, 2, 4, 8, 16}.

Usage:
    uv run python scripts/bench_parallel.py
"""

from __future__ import annotations

import time

import mlx.core as mx

from babbling_baby.pupil import Pupil
from babbling_baby.train import train_one_step


CONTEXT = b"hello "
TARGET = b"world "
STEPS = 100
CANDIDATES = [1, 2, 4, 8, 16]


def time_n_pupils(n: int, steps: int = STEPS) -> float:
    """Train N pupils for `steps` iterations each. All forward/
    backward calls are submitted in sequence, then we mx.eval() once
    at the end of each step to synchronise. If MLX's lazy scheduler
    can overlap independent model operations, the time will scale
    sub-linearly with N."""
    pupils = [Pupil() for _ in range(n)]
    # Warm-up step so first iteration's compile cost isn't counted
    for p in pupils:
        train_one_step(p, CONTEXT, TARGET)
    mx.eval(*[p.parameters() for p in pupils])

    start = time.perf_counter()
    for _ in range(steps):
        for p in pupils:
            train_one_step(p, CONTEXT, TARGET)
    elapsed = time.perf_counter() - start
    return elapsed


def main() -> None:
    print(f"benchmark: {STEPS} training steps per pupil, batch size 1, no sampling overhead")
    print()
    print(f"{'N pupils':>10}  {'time (s)':>10}  {'s/step total':>14}  {'s/step per pupil':>17}  {'speedup vs N=1':>14}")
    print("-" * 80)

    baseline = None
    for n in CANDIDATES:
        elapsed = time_n_pupils(n)
        per_step = elapsed / STEPS
        per_step_per_pupil = per_step / n
        if baseline is None:
            baseline = per_step_per_pupil
            speedup = "—"
        else:
            speedup = f"{baseline / per_step_per_pupil:.2f}x"
        print(f"{n:>10}  {elapsed:>10.2f}  {per_step:>14.4f}  {per_step_per_pupil:>17.5f}  {speedup:>14}")


if __name__ == "__main__":
    main()

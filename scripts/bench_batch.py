"""Benchmark how training step time scales with the batch dimension
within a single model. If the GPU is compute-unsaturated for batch=1,
larger batches should give close-to-flat per-example time."""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from babbling_baby.pupil import Pupil, VOCAB_SIZE


CONTEXT = b"hello "
TARGET = b"world "
STEPS = 100
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def batched_loss(pupil: Pupil, batch_size: int) -> mx.array:
    """Run a forward+loss over a stacked batch of identical
    (context, target) pairs. The point isn't the data — it's to
    expose compute scaling with batch dim."""
    full = list(CONTEXT + TARGET)
    # shape (B, seq_len)
    tokens = mx.array([full] * batch_size, dtype=mx.int32)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits = pupil(inputs)  # (B, seq_len-1, vocab)
    n_context = len(CONTEXT)
    n_target = len(TARGET)
    start = max(0, n_context - 1)
    target_logits = logits[:, start:start + n_target, :]
    target_tokens = targets[:, start:start + n_target]
    return nn.losses.cross_entropy(
        target_logits.reshape(-1, VOCAB_SIZE),
        target_tokens.reshape(-1),
        reduction="mean",
    )


def time_batch(batch_size: int, steps: int = STEPS) -> float:
    pupil = Pupil()
    optimizer = optim.AdamW(learning_rate=3e-4)

    def loss_fn(model: Pupil) -> mx.array:
        return batched_loss(model, batch_size)

    grad_fn = nn.value_and_grad(pupil, loss_fn)

    # Warm-up
    loss, grads = grad_fn(pupil)
    optimizer.update(pupil, grads)
    mx.eval(pupil.parameters(), optimizer.state)

    start = time.perf_counter()
    for _ in range(steps):
        loss, grads = grad_fn(pupil)
        optimizer.update(pupil, grads)
        mx.eval(pupil.parameters(), optimizer.state)
    return time.perf_counter() - start


def main() -> None:
    print(f"benchmark: {STEPS} training steps per config")
    print()
    print(f"{'batch':>6}  {'time (s)':>10}  {'s/step':>10}  {'examples/sec':>14}  {'rel /ex vs B=1':>16}")
    print("-" * 70)

    baseline = None
    for B in BATCH_SIZES:
        try:
            elapsed = time_batch(B)
        except Exception as e:
            print(f"{B:>6}  failed: {e}")
            continue
        per_step = elapsed / STEPS
        ex_per_sec = (STEPS * B) / elapsed
        if baseline is None:
            baseline = per_step / B
            ratio = "—"
        else:
            ratio = f"{(per_step / B) / baseline:.3f}"
        print(f"{B:>6}  {elapsed:>10.2f}  {per_step:>10.4f}  {ex_per_sec:>14.0f}  {ratio:>16}")


if __name__ == "__main__":
    main()

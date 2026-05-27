"""Interactive training loop. The pupil learns to predict the
teacher's next emission given the partner's last word, by standard
next-byte cross-entropy."""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from babbling_baby.pupil import Pupil, VOCAB_SIZE, sample_emissions
from babbling_baby.trace import all_metrics


_optimizers: dict[int, optim.Optimizer] = {}


def _get_optimizer(pupil: Pupil, lr: float = 3e-4) -> optim.Optimizer:
    key = id(pupil)
    if key not in _optimizers:
        _optimizers[key] = optim.AdamW(learning_rate=lr)
    return _optimizers[key]


def sequence_loss(pupil: Pupil, context: bytes, target: bytes) -> mx.array:
    """Cross-entropy of producing `target` byte-by-byte after `context`.

    Only positions whose prediction lies within the target are
    counted in the loss — the context is conditioning, not a
    prediction target.
    """
    if len(target) == 0:
        return mx.array(0.0)
    full = bytes(context) + bytes(target)
    cfg = pupil.config
    if len(full) > cfg.context_len:
        drop = len(full) - cfg.context_len
        full = full[drop:]
        context_len_eff = max(0, len(context) - drop)
    else:
        context_len_eff = len(context)
    if len(full) < 2:
        return mx.array(0.0)
    tokens = mx.array([list(full)], dtype=mx.int32)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits = pupil(inputs)
    start = max(0, context_len_eff - 1)
    n_target = len(target)
    end = min(start + n_target, targets.shape[1])
    target_logits = logits[:, start:end, :]
    target_tokens = targets[:, start:end]
    loss = nn.losses.cross_entropy(
        target_logits.reshape(-1, VOCAB_SIZE),
        target_tokens.reshape(-1),
        reduction="mean",
    )
    return loss


def train_one_step(pupil: Pupil, context: bytes, target: bytes) -> float:
    def loss_fn(model: Pupil) -> mx.array:
        return sequence_loss(model, context, target)

    loss_and_grad_fn = nn.value_and_grad(pupil, loss_fn)
    loss, grads = loss_and_grad_fn(pupil)
    optimizer = _get_optimizer(pupil)
    optimizer.update(pupil, grads)
    mx.eval(pupil.parameters(), optimizer.state)
    return float(loss)


def rl_train_one_step(
    pupil: Pupil,
    context: bytes,
    sampled: bytes,
    reward: float,
) -> float:
    """One REINFORCE-style step on a sampled emission.

    The gradient minimises `reward * sequence_loss(pupil, context, sampled)`.
    Positive reward pushes the pupil toward the sampled emission (the
    same direction as imitation training on it). Negative reward pushes
    away — the pupil becomes less likely to emit it.

    Mathematically equivalent to the REINFORCE policy gradient
    `-reward * log_pi(sampled | context)` when the sequence loss is
    the per-token mean negative log likelihood: sequence_loss returns
    -log_pi / n, and we treat the per-token scale as a constant factor
    absorbed into the learning rate.
    """
    if len(sampled) == 0 or reward == 0.0:
        return 0.0

    def loss_fn(model: Pupil) -> mx.array:
        return reward * sequence_loss(model, context, sampled)

    loss_and_grad_fn = nn.value_and_grad(pupil, loss_fn)
    loss, grads = loss_and_grad_fn(pupil)
    optimizer = _get_optimizer(pupil)
    optimizer.update(pupil, grads)
    mx.eval(pupil.parameters(), optimizer.state)
    return float(loss)


def sample_word_until_delimiter(
    pupil: Pupil,
    context: bytes,
    max_len: int = 16,
    temperature: float = 1.0,
) -> bytes:
    """Sample bytes from the pupil until a space or until max_len.
    Includes the trailing space if one was sampled within max_len."""
    cfg = pupil.config
    seq = list(context[-(cfg.context_len - 1) :]) if context else [ord(" ")]
    out: list[int] = []
    delimiter = ord(" ")
    for _ in range(max_len):
        window = seq[-cfg.context_len :] or [delimiter]
        tokens = mx.array([window], dtype=mx.int32)
        logits = pupil(tokens)
        last_logits = logits[0, -1] / max(temperature, 1e-6)
        probs = mx.softmax(last_logits)
        token = int(mx.random.categorical(mx.log(probs + 1e-9)).item())
        out.append(token)
        seq.append(token)
        if token == delimiter:
            break
    return bytes(out)


WORD_DELIMITER = b" "
REWARD_REAL_WORD = 1.0
# Penalty for non-words is shaped by how far the emission's length is
# from the teacher's typical word length. Near-mean length gets a mild
# penalty (-0.1); a 1-char fragment gets close to -1.0.
NON_WORD_BASE_PENALTY = 0.1
NON_WORD_MAX_PENALTY = 1.0


@functools.cache
def _teacher_length_stats() -> tuple[float, float]:
    """Mean and stddev of teacher word lengths. Computed once from the
    verifier's real-word list since the teacher draws from that set."""
    from babbling_baby.verifier import _real_word_list

    lengths = [len(w) for w in _real_word_list()]
    n = len(lengths)
    mean = sum(lengths) / n
    var = sum((L - mean) ** 2 for L in lengths) / n
    return mean, max(var ** 0.5, 1.0)


def compute_reward(emission: bytes, is_real: bool) -> float:
    """Length-aware reward.

    - Real words score the full positive reward regardless of length.
    - Non-words score a penalty whose magnitude scales with how far
      the emission's length is from the teacher's typical word length.
      A non-word at the mean length gets the base penalty (-0.1); a
      non-word that's many stddevs short or long approaches -1.0.

    The trailing space delimiter is stripped before measuring length.
    """
    word = emission.rstrip(WORD_DELIMITER)
    if is_real:
        return REWARD_REAL_WORD
    mean_len, std_len = _teacher_length_stats()
    deviation_in_sigmas = abs(len(word) - mean_len) / std_len
    # 0 stddevs out: base penalty (-0.1). 2+ stddevs out: max penalty (-1.0).
    extra = (NON_WORD_MAX_PENALTY - NON_WORD_BASE_PENALTY) * min(deviation_in_sigmas / 2.0, 1.0)
    return -(NON_WORD_BASE_PENALTY + extra)


def train(
    time_budget_seconds: int,
    log_interval_s: float | None = None,
    run_dir: Path | str | None = None,
    sample_n_bytes: int = 512,
    rl_enabled: bool = True,
) -> Pupil:
    """Run interactive training for up to time_budget_seconds.

    Each iteration does two updates:

    1. Imitation: train the pupil to predict the teacher's delimited
       word given the previous delimited word as context.
    2. RL: the pupil samples its own word; the curriculum's current
       vocabulary scores it; a REINFORCE-style step pulls the pupil
       toward scoring emissions.

    A length-based curriculum scopes the teacher and the scoring. The
    pupil first faces single-letter words, then two-letter words, then
    progressively longer real words as it succeeds.

    If `run_dir` and `log_interval_s` are provided, trace metrics are
    sampled from the pupil at that cadence and appended to
    `<run_dir>/trace.jsonl`.
    """
    from babbling_baby.curriculum import Curriculum
    from babbling_baby.teacher import DictionaryTeacher

    curriculum = Curriculum()
    pupil = Pupil()
    teacher = DictionaryTeacher(curriculum=curriculum)
    last_word = curriculum.random_word().encode("utf-8") + WORD_DELIMITER

    log_file = None
    if run_dir is not None and log_interval_s is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = (run_dir / "trace.jsonl").open("w")

    start = time.time()
    deadline = start + time_budget_seconds
    last_log = start
    loss_ema: float | None = None
    steps = 0
    level_history: list[tuple[float, int]] = [(0.0, curriculum.level)]

    while time.time() < deadline:
        teacher_emission = teacher.emit(context=last_word) + WORD_DELIMITER
        loss = train_one_step(pupil, last_word, teacher_emission)
        loss_ema = loss if loss_ema is None else 0.95 * loss_ema + 0.05 * loss

        if rl_enabled:
            sampled = sample_word_until_delimiter(pupil, context=last_word, max_len=16)
            sampled_word = sampled.rstrip(WORD_DELIMITER)
            if sampled_word:
                matched_level, hit = curriculum.score(sampled_word)
            else:
                matched_level, hit = None, False
            # Positive-only reward: +1 for hit, 0 for miss. The
            # curriculum provides the difficulty gradient; we don't
            # need to shape penalties on top.
            if hit:
                rl_train_one_step(pupil, last_word, sampled_word, reward=1.0)
            if sampled_word:
                advanced = curriculum.record(sampled_word)
            else:
                advanced = curriculum.record(b"")
            if advanced:
                level_history.append((time.time() - start, curriculum.level))

        last_word = teacher_emission
        steps += 1

        if log_file is not None:
            now = time.time()
            if now - last_log >= log_interval_s:
                sample = sample_emissions(
                    pupil,
                    n_bytes=sample_n_bytes,
                    context=curriculum.random_word().encode("utf-8") + WORD_DELIMITER,
                )
                metrics = all_metrics(sample)
                point = {
                    "step": steps,
                    "elapsed_s": round(now - start, 2),
                    "loss_ema": round(loss_ema, 4),
                    "level": curriculum.level,
                    "level_name": curriculum.level_name,
                    "level_success_rate": round(curriculum.success_rate(), 4),
                    "per_level_rate": [
                        round(curriculum.success_rate(L), 4)
                        for L in range(len(curriculum.LEVELS))
                    ],
                    **{k: round(v, 4) for k, v in metrics.items()},
                    "sample_preview": sample[:128].decode("latin-1", errors="replace"),
                }
                log_file.write(json.dumps(point) + "\n")
                log_file.flush()
                last_log = now

    if log_file is not None:
        log_file.close()

    if run_dir is not None:
        level_path = Path(run_dir) / "levels.jsonl"
        with level_path.open("w") as f:
            for t, lvl in level_history:
                f.write(json.dumps({"elapsed_s": round(t, 2), "level": lvl}) + "\n")

    return pupil

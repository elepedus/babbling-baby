"""Training: one step should bias the pupil toward the teacher's
emission. After enough steps, the pupil's emissions should look like
the teacher's (real words)."""

import mlx.core as mx

from babbling_baby.pupil import Pupil
from babbling_baby.train import sequence_loss, train_one_step


def test_one_step_decreases_loss_on_target():
    """A single training step on (context, target) should reduce the
    cross-entropy loss of that target — i.e., the model becomes more
    likely to produce the target."""
    mx.random.seed(0)
    pupil = Pupil()
    context = b"hello"
    target = b" world"

    loss_before = float(sequence_loss(pupil, context, target))
    train_one_step(pupil, context, target)
    loss_after = float(sequence_loss(pupil, context, target))

    assert loss_after < loss_before, (
        f"Loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}"
    )


def test_repeated_steps_reduce_loss_meaningfully():
    """Many steps on the same target should drive loss down close to
    zero, confirming the optimiser actually fits."""
    mx.random.seed(0)
    pupil = Pupil()
    context = b"the"
    target = b" quick"

    loss_initial = float(sequence_loss(pupil, context, target))
    for _ in range(50):
        train_one_step(pupil, context, target)
    loss_final = float(sequence_loss(pupil, context, target))

    assert loss_final < loss_initial * 0.5, (
        f"50 steps barely reduced loss: {loss_initial:.4f} -> {loss_final:.4f}"
    )


def test_brief_training_produces_word_boundaries():
    """After a few seconds of training, the pupil's emissions should
    include non-letter bytes (i.e., word boundaries). The training
    data is space-delimited so the pupil sees boundaries and learns
    to emit them. If this fails, the pupil is producing one continuous
    letter stream with no boundaries — symptom of missing delimiters
    in the training protocol."""
    from babbling_baby.train import train as run_training
    from babbling_baby.pupil import sample_emissions as sample
    from babbling_baby.verifier import random_word

    pupil = run_training(time_budget_seconds=5)
    seed = random_word().encode("utf-8") + b" "
    emissions = sample(pupil, n_bytes=512, context=seed)
    non_letter_bytes = sum(
        1 for b in emissions if not ((65 <= b <= 90) or (97 <= b <= 122))
    )
    assert non_letter_bytes > 0, (
        f"pupil emitted no word boundaries in 512 bytes after training; "
        f"sample: {emissions[:128]!r}"
    )


def test_rl_step_with_positive_reward_decreases_loss_on_sample():
    """A positive reward should push the pupil toward the sampled
    emission — the same direction as imitation training on that
    emission. After the RL step, sequence_loss on (context, sampled)
    should be lower."""
    from babbling_baby.train import rl_train_one_step

    mx.random.seed(0)
    pupil = Pupil()
    context = b"hello "
    sampled = b"world "

    loss_before = float(sequence_loss(pupil, context, sampled))
    rl_train_one_step(pupil, context, sampled, reward=1.0)
    loss_after = float(sequence_loss(pupil, context, sampled))

    assert loss_after < loss_before, (
        f"positive reward did not decrease loss on sampled emission: "
        f"{loss_before:.4f} -> {loss_after:.4f}"
    )


def test_rl_step_with_negative_reward_increases_loss_on_sample():
    """A negative reward should push the pupil away from the sampled
    emission. After the RL step, sequence_loss on (context, sampled)
    should be higher — the pupil becomes less likely to produce it."""
    from babbling_baby.train import rl_train_one_step

    mx.random.seed(0)
    pupil = Pupil()
    context = b"hello "
    sampled = b"xqzv "

    loss_before = float(sequence_loss(pupil, context, sampled))
    rl_train_one_step(pupil, context, sampled, reward=-1.0)
    loss_after = float(sequence_loss(pupil, context, sampled))

    assert loss_after > loss_before, (
        f"negative reward did not increase loss on sampled emission: "
        f"{loss_before:.4f} -> {loss_after:.4f}"
    )


def test_sample_word_until_delimiter_returns_bytes():
    """Sampling a single word from the pupil yields bytes; if the
    pupil emits a space within the limit, the word ends there."""
    from babbling_baby.train import sample_word_until_delimiter

    mx.random.seed(0)
    pupil = Pupil()
    word = sample_word_until_delimiter(pupil, context=b"hello ", max_len=16)
    assert isinstance(word, bytes)
    assert 1 <= len(word) <= 16


def test_reward_real_word_is_full_positive():
    """A real dictionary word gets the full positive reward regardless
    of its length — it's already a real word; that's what we want."""
    from babbling_baby.train import compute_reward

    assert compute_reward(b"cat", is_real=True) == 1.0
    assert compute_reward(b"magnificent", is_real=True) == 1.0


def test_reward_non_word_typical_length_is_mild_penalty():
    """A non-word whose length is close to the teacher's typical word
    length gets only a small penalty — it's nearly a word, just wrong.
    The teacher's mean word length is around 8 chars."""
    from babbling_baby.train import compute_reward

    # 8-char non-word: at the mean
    r = compute_reward(b"xyzqrabt", is_real=False)
    assert -0.2 <= r <= 0, f"expected small penalty near typical length, got {r}"


def test_reward_non_word_far_from_typical_is_strong_penalty():
    """A 1- or 2-char non-word is many stddevs below typical and
    should get a much larger penalty than a length-typical non-word."""
    from babbling_baby.train import compute_reward

    r_short = compute_reward(b"s", is_real=False)
    r_typical = compute_reward(b"xyzqrabt", is_real=False)
    assert r_short < r_typical, (
        f"short non-word should be penalised more than typical-length "
        f"non-word: short={r_short:.3f} typical={r_typical:.3f}"
    )
    assert r_short <= -0.5, f"expected strong penalty for 1-char non-word, got {r_short}"


def test_reward_stripped_of_trailing_delimiter():
    """The reward considers the word content, not the trailing space."""
    from babbling_baby.train import compute_reward

    r_with_space = compute_reward(b"cat ", is_real=True)
    r_without_space = compute_reward(b"cat", is_real=True)
    assert r_with_space == r_without_space

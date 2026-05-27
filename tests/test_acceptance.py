"""Proxy acceptance test for Babbling Baby v0.

After roughly 1 minute of interactive training, the pupil's emission
stream should be more than 50% printable ASCII letters [a-zA-Z]. This
is faster and more interpretable than the real acceptance test
(dictionary words climbing meaningfully above the untrained baseline),
but faithful enough to drive the outside-in BDD loop.

This is the one test that exists initially. Everything else is
discovered by running it and watching it fail.
"""

from babbling_baby.train import train
from babbling_baby.pupil import sample_emissions


def percent_letters(byte_stream: bytes) -> float:
    if not byte_stream:
        return 0.0
    letters = sum(1 for b in byte_stream if (65 <= b <= 90) or (97 <= b <= 122))
    return letters / len(byte_stream)


def test_pupil_babbles_letters_after_one_minute():
    pupil = train(time_budget_seconds=60)
    emissions = sample_emissions(pupil, n_bytes=1000)
    assert percent_letters(emissions) > 0.5

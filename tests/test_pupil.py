"""The pupil is a byte-level decoder-only transformer with random
initialisation. It can be sampled to produce byte sequences."""

from babbling_baby.pupil import Pupil, sample_emissions


def test_pupil_constructs():
    pupil = Pupil()
    assert pupil is not None


def test_pupil_samples_bytes():
    """Sampling from a freshly-initialised pupil should produce bytes
    of the requested length."""
    pupil = Pupil()
    emissions = sample_emissions(pupil, n_bytes=128)
    assert isinstance(emissions, bytes)
    assert len(emissions) == 128


def test_pupil_samples_vary_between_calls():
    """Two consecutive samples from the same untrained pupil should
    differ. (Random sampling, not argmax, so this should hold.)"""
    pupil = Pupil()
    a = sample_emissions(pupil, n_bytes=128)
    b = sample_emissions(pupil, n_bytes=128)
    assert a != b


def test_pupil_can_take_context():
    """Sampling should accept an optional context (the partner's last
    word) and condition on it."""
    pupil = Pupil()
    emissions = sample_emissions(pupil, n_bytes=64, context=b"hello")
    assert isinstance(emissions, bytes)
    assert len(emissions) == 64

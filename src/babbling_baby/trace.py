"""Four-granularity progress trace. Lets us watch learning happen
at multiple levels, not just at the final dictionary-word level:

  1. %printable_ascii        — is the model emitting bytes humans can see?
  2. %letters                — is it emitting letters specifically?
  3. %plausible_word_shape   — are letter-runs of word-like length?
  4. %dictionary_words       — are those runs actual real words?

The progression from (1) to (4) is how the pupil grows up in
miniature. Watching all four during training shows *how* learning
happens, not only whether it does.
"""

from __future__ import annotations

from babbling_baby.verifier import is_word


SHAPE_MIN_LEN = 3
SHAPE_MAX_LEN = 12


def _is_letter(b: int) -> bool:
    return (65 <= b <= 90) or (97 <= b <= 122)


def _letter_runs(stream: bytes) -> list[bytes]:
    runs: list[bytes] = []
    current: list[int] = []
    for b in stream:
        if _is_letter(b):
            current.append(b)
        elif current:
            runs.append(bytes(current))
            current = []
    if current:
        runs.append(bytes(current))
    return runs


def percent_printable_ascii(stream: bytes) -> float:
    if not stream:
        return 0.0
    printable = sum(1 for b in stream if 32 <= b <= 126)
    return printable / len(stream)


def percent_letters(stream: bytes) -> float:
    if not stream:
        return 0.0
    return sum(1 for b in stream if _is_letter(b)) / len(stream)


def percent_plausible_word_shape(stream: bytes) -> float:
    runs = _letter_runs(stream)
    if not runs:
        return 0.0
    well_sized = sum(1 for r in runs if SHAPE_MIN_LEN <= len(r) <= SHAPE_MAX_LEN)
    return well_sized / len(runs)


def percent_dictionary_words(stream: bytes) -> float:
    runs = _letter_runs(stream)
    if not runs:
        return 0.0
    real = sum(1 for r in runs if is_word(r))
    return real / len(runs)


def all_metrics(stream: bytes) -> dict[str, float]:
    """Compute all four trace metrics in one pass-friendly call."""
    return {
        "percent_printable": percent_printable_ascii(stream),
        "percent_letters": percent_letters(stream),
        "percent_shape": percent_plausible_word_shape(stream),
        "percent_dict": percent_dictionary_words(stream),
    }

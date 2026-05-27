"""Graded adaptive length-based curriculum with overlapping levels.

At any point, the teacher draws from a mixture of levels — the
*focus* level plus its immediate neighbours (focus−1 and focus+1).
The mixture weights favour the focus level (70%) with maintenance
on the previous level (20%) and an early easy shot at the next
level (10%).

Each level has its own unlock threshold, decreasing with difficulty:
mastery of single-letter words must be reliable (85%); mastery of
long words just needs a meaningful foothold (10%). When the focus
level's recent success rate meets its own threshold, focus advances
and the previous level continues to receive maintenance practice.

This is the natural shape of a developmental curriculum: foundational
skills are kept solid while harder skills are explored — the pupil
gets early exposure to the next rung before being required to master
it, and easier rungs keep being practised long after they're learned.
"""

from __future__ import annotations

import random
from collections import deque
from functools import cache

ONE_LETTER_WORDS: tuple[str, ...] = ("a", "i")
TWO_LETTER_WORDS: tuple[str, ...] = (
    "an", "as", "at", "be", "by", "do", "go", "he", "hi", "if",
    "in", "is", "it", "me", "my", "no", "of", "oh", "on", "or",
    "so", "to", "up", "us", "we",
)


@cache
def _strict_words_by_length() -> dict[int, tuple[str, ...]]:
    from babbling_baby.verifier import _real_word_list

    buckets: dict[int, list[str]] = {}
    for w in _real_word_list():
        buckets.setdefault(len(w), []).append(w)
    return {k: tuple(v) for k, v in buckets.items()}


def _strict_words_in_range(min_len: int, max_len: int) -> tuple[str, ...]:
    out: list[str] = []
    by_len = _strict_words_by_length()
    for L in range(min_len, max_len + 1):
        out.extend(by_len.get(L, ()))
    return tuple(out)


def _all_strict_words() -> tuple[str, ...]:
    from babbling_baby.verifier import _real_word_list
    return _real_word_list()


class Curriculum:
    """Graded length-based adaptive curriculum.

    Levels:
      0: single letters (a, i)
      1: two-letter common words
      2: three-letter strict words
      3: 4–5 letter strict words
      4: 6–8 letter strict words
      5: full strict dictionary
    """

    LEVELS: tuple[tuple[str, object], ...] = (
        ("one_letter", ONE_LETTER_WORDS),
        ("two_letter", TWO_LETTER_WORDS),
        ("three_letter", (3, 3)),
        ("four_to_five", (4, 5)),
        ("six_to_eight", (6, 8)),
        ("full_strict", "all"),
    )

    # Per-level unlock threshold — decreasing with difficulty.
    UNLOCK_THRESHOLDS: tuple[float, ...] = (0.85, 0.65, 0.45, 0.30, 0.20, 0.10)

    PROMOTION_WINDOW = 100

    # Mixture weights when sampling from the active level window.
    WEIGHT_FOCUS = 0.70
    WEIGHT_PREV = 0.20
    WEIGHT_NEXT = 0.10

    def __init__(self) -> None:
        self.focus = 0
        self._success_windows: list[deque[bool]] = [
            deque(maxlen=self.PROMOTION_WINDOW) for _ in self.LEVELS
        ]
        self._vocab_lists: list[tuple[str, ...]] = []
        self._vocab_sets: list[frozenset[str]] = []
        for spec in (level_spec for _name, level_spec in self.LEVELS):
            if isinstance(spec, tuple) and all(isinstance(x, str) for x in spec):
                words: tuple[str, ...] = spec  # type: ignore[assignment]
            elif spec == "all":
                words = _all_strict_words()
            else:
                min_len, max_len = spec  # type: ignore[misc]
                words = _strict_words_in_range(min_len, max_len)
            self._vocab_lists.append(words)
            self._vocab_sets.append(frozenset(words))

    # --- compatibility surface ---

    @property
    def level(self) -> int:
        return self.focus

    @property
    def level_name(self) -> str:
        return self.LEVELS[self.focus][0]

    @property
    def vocab(self) -> frozenset[str]:
        """Vocab at the current focus level. (Kept for tests/inspection.)"""
        return self._vocab_sets[self.focus]

    # --- level set ---

    def active_levels(self) -> tuple[int, ...]:
        """The levels currently in the teacher's mixture: focus and its
        immediate neighbours."""
        out = [self.focus]
        if self.focus > 0:
            out.insert(0, self.focus - 1)
        if self.focus < len(self.LEVELS) - 1:
            out.append(self.focus + 1)
        return tuple(out)

    def _level_weight(self, level: int) -> float:
        if level == self.focus:
            return self.WEIGHT_FOCUS
        if level == self.focus - 1:
            return self.WEIGHT_PREV
        if level == self.focus + 1:
            return self.WEIGHT_NEXT
        return 0.0

    def sample_level(self) -> int:
        active = self.active_levels()
        weights = [self._level_weight(L) for L in active]
        return random.choices(active, weights=weights, k=1)[0]

    # --- vocab access ---

    def random_word(self) -> str:
        level = self.sample_level()
        return random.choice(self._vocab_lists[level])

    def score(self, emission: str | bytes) -> tuple[int | None, bool]:
        """Returns (matching_level, hit). Checks every active level;
        an emission is at most at one level since vocabs are disjoint
        by length."""
        if isinstance(emission, bytes):
            try:
                emission = emission.decode("utf-8")
            except UnicodeDecodeError:
                return None, False
        if not emission:
            return None, False
        word = emission.lower()
        for L in self.active_levels():
            if word in self._vocab_sets[L]:
                return L, True
        return None, False

    # --- recording and promotion ---

    def record(self, emission: str | bytes) -> bool:
        """Score the pupil's emission and update per-level windows.

        Each active level's window records True iff the pupil hit at
        *that* level (not just any level). The focus level's window is
        what governs promotion.

        Returns True if focus advanced as a result.
        """
        scored_level, hit = self.score(emission)
        for L in self.active_levels():
            self._success_windows[L].append(hit and scored_level == L)
        return self._check_advance()

    def _check_advance(self) -> bool:
        if self.focus >= len(self.LEVELS) - 1:
            return False
        window = self._success_windows[self.focus]
        if len(window) < self.PROMOTION_WINDOW:
            return False
        rate = sum(window) / len(window)
        threshold = self.UNLOCK_THRESHOLDS[self.focus]
        if rate >= threshold:
            self.focus += 1
            # New focus has to earn its own track record from scratch.
            self._success_windows[self.focus].clear()
            return True
        return False

    def success_rate(self, level: int | None = None) -> float:
        if level is None:
            level = self.focus
        w = self._success_windows[level]
        return (sum(w) / len(w)) if w else 0.0

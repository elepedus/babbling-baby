"""The teacher emits real words on demand.

For v0 the teacher is a fast dictionary sampler driven by an adaptive
curriculum. The curriculum determines which word-length range the
teacher draws from; the teacher itself is just a sampler.

The spec's only hard requirement is 'produces real words competently';
a uniform sampler from the current curriculum level's wordlist
satisfies that with zero API latency, which is what makes the BDD
loop tractable.

A Claude-backed teacher (or any other LLM) can be swapped in later
behind the same `.emit(context)` interface for richer experiments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from babbling_baby.curriculum import Curriculum


class DictionaryTeacher:
    """A teacher that emits a random dictionary word per turn, drawn
    from the curriculum's current-level vocabulary. Ignores context."""

    def __init__(self, curriculum: Curriculum | None = None) -> None:
        if curriculum is None:
            from babbling_baby.curriculum import Curriculum as _Curriculum
            curriculum = _Curriculum()
        self.curriculum = curriculum

    def emit(self, context: bytes) -> bytes:
        return self.curriculum.random_word().encode("utf-8")

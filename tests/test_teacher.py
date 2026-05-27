"""The teacher emits words on demand, drawing from the curriculum's
active level mixture (focus +/- 1). For v0 the teacher is a fast
dictionary sampler driven by an adaptive graded curriculum."""

from babbling_baby.curriculum import Curriculum
from babbling_baby.teacher import DictionaryTeacher


def test_teacher_emits_a_word_recognised_by_its_curriculum():
    curriculum = Curriculum()
    teacher = DictionaryTeacher(curriculum=curriculum)
    word = teacher.emit(context=b"hello").decode("utf-8")
    _, scored = curriculum.score(word)
    assert scored


def test_teacher_emits_bytes():
    teacher = DictionaryTeacher()
    word = teacher.emit(context=b"hello")
    assert isinstance(word, bytes)


def test_teacher_at_focus_zero_mostly_emits_one_letter_words():
    """At focus 0 the active mixture is levels 0 and 1. Most emissions
    should be at level 0 (1-letter); some at level 1 (2-letter) for
    early stretch."""
    curriculum = Curriculum()
    teacher = DictionaryTeacher(curriculum=curriculum)
    lengths = [len(teacher.emit(context=b"")) for _ in range(200)]
    one_letter_share = sum(1 for L in lengths if L == 1) / len(lengths)
    # Focus weight is 0.70, prev does not exist at focus 0, next is 0.10
    # → focus should be ~88%, next ~12% (after normalising)
    assert one_letter_share > 0.6, (
        f"expected >60% one-letter emissions at focus 0, got {one_letter_share:.2%}"
    )


def test_teacher_at_focus_two_includes_neighbouring_lengths():
    """At focus 2 (3-letter words) the mixture includes levels 1, 2, 3
    so lengths should span 2..5."""
    curriculum = Curriculum()
    curriculum.focus = 2
    teacher = DictionaryTeacher(curriculum=curriculum)
    lengths = {len(teacher.emit(context=b"")) for _ in range(200)}
    # Should see lengths from neighbouring levels: 2 (prev), 3 (focus),
    # 4 or 5 (next from four_to_five level)
    assert 3 in lengths
    assert lengths & {2, 4, 5}


def test_teacher_emit_accepts_empty_context():
    curriculum = Curriculum()
    teacher = DictionaryTeacher(curriculum=curriculum)
    word = teacher.emit(context=b"")
    _, scored = curriculum.score(word)
    assert scored

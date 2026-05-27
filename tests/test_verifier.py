"""The verifier is a dictionary lookup. It returns True for real
English words, False otherwise. No learned judgement, no ambiguity."""

from babbling_baby.verifier import is_word


def test_recognises_common_words():
    assert is_word("hello")
    assert is_word("cat")
    assert is_word("magnificent")


def test_rejects_non_words():
    assert not is_word("xyzq")
    assert not is_word("qqqqq")


def test_rejects_short_letter_combinations():
    """Letter combos that appear in some wordlists as abbreviations or
    chemical symbols but aren't 'real meaningful dictionary words' to
    a native speaker. The stricter criterion (length>=3 + WordNet
    synset) eliminates these.

    sis (informal for sister) and spry (real adjective) are fine. Plain
    letter clusters are not."""
    assert not is_word("se")
    assert not is_word("sh")
    assert not is_word("ss")
    assert not is_word("st")
    assert not is_word("sm")
    assert not is_word("sta")
    assert not is_word("ka")
    assert not is_word("lo")


def test_accepts_short_real_words():
    """Real meaningful words that are short — three letters with
    actual semantic content — should still pass."""
    assert is_word("sis")
    assert is_word("spry")
    assert is_word("cat")
    assert is_word("dog")
    assert is_word("tor")


def test_rejects_empty():
    assert not is_word("")


def test_case_insensitive():
    assert is_word("Hello")
    assert is_word("HELLO")


def test_accepts_bytes_input():
    """Pupil emits bytes; verifier should accept either bytes or str."""
    assert is_word(b"hello")
    assert not is_word(b"xyzq")


def test_random_word_returns_a_real_word():
    """Game initialisation seeds with a random dictionary word."""
    from babbling_baby.verifier import random_word

    for _ in range(20):
        word = random_word()
        assert is_word(word), f"random_word produced non-word: {word!r}"


def test_random_word_returns_varied_words():
    """Should not always return the same word."""
    from babbling_baby.verifier import random_word

    samples = {random_word() for _ in range(50)}
    assert len(samples) > 1

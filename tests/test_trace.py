"""Progress trace: four metrics computed on any byte stream.
Used to watch learning happen at multiple granularities, not only
at the final dictionary-word level."""

from babbling_baby.trace import (
    percent_dictionary_words,
    percent_letters,
    percent_plausible_word_shape,
    percent_printable_ascii,
)


# --- percent_printable_ascii ---


def test_printable_zero_on_null_bytes():
    assert percent_printable_ascii(b"\x00\x01\x02") == 0.0


def test_printable_one_on_all_letters():
    assert percent_printable_ascii(b"hello world") == 1.0


def test_printable_empty_returns_zero():
    assert percent_printable_ascii(b"") == 0.0


# --- percent_letters ---


def test_letters_one_on_all_letters():
    assert percent_letters(b"helloworld") == 1.0


def test_letters_zero_on_digits():
    assert percent_letters(b"1234567890") == 0.0


def test_letters_half_on_mixed():
    # 5 letters, 5 spaces -> 0.5
    assert percent_letters(b"abcde     ") == 0.5


# --- percent_plausible_word_shape ---


def test_shape_one_on_well_sized_word_runs():
    # Three letter-runs, all 3-12 chars
    assert percent_plausible_word_shape(b"cat dog mouse") == 1.0


def test_shape_zero_on_single_letters():
    # Letter-runs are all length 1, below the 3-char threshold
    assert percent_plausible_word_shape(b"a b c d e f") == 0.0


def test_shape_excludes_runs_longer_than_12():
    # One short run, one too-long run
    assert percent_plausible_word_shape(b"cat thisismuchtoolongtobeaword") == 0.5


def test_shape_zero_on_empty():
    assert percent_plausible_word_shape(b"") == 0.0


# --- percent_dictionary_words ---


def test_dict_one_on_real_words():
    # Three real words, separated by spaces
    assert percent_dictionary_words(b"cat dog mouse") == 1.0


def test_dict_zero_on_gibberish():
    assert percent_dictionary_words(b"xyzq qzyrk zzqq") == 0.0


def test_dict_partial_on_mix():
    # 2 real + 2 fake -> 0.5
    assert percent_dictionary_words(b"cat xyzq dog qzyrk") == 0.5


def test_dict_zero_on_empty():
    assert percent_dictionary_words(b"") == 0.0

"""Dictionary lookup verifier. The judge of whether a byte sequence
forms a real, meaningful English word.

The criterion: length >= 3 AND has at least one WordNet synset.

This is stricter than a plain wordlist membership check because
wordlists (pyspellchecker, /usr/share/dict/words, etc.) contain many
letter-combos that aren't 'real words' — chemical symbols, compass
abbreviations, single-letter unit symbols, Nazi initialisms, and so
on. A native speaker would not call 'se', 'ss', 'st', or 'sm' words.
Requiring a WordNet synset filters most of that out while keeping
genuinely short content words like 'sis', 'spry', 'cat', 'tor'.

Function words like 'the' and 'and' are excluded by this criterion
(they don't have WordNet synsets). That's acceptable for the babbling
experiment — the bar is whether the pupil emits content-bearing
words, not whether it emits common articles."""

import random
from functools import cache

import nltk
from nltk.corpus import wordnet
from spellchecker import SpellChecker


MIN_WORD_LEN = 3


def _ensure_wordnet() -> None:
    try:
        wordnet.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


@cache
def _checker() -> SpellChecker:
    return SpellChecker(language="en")


@cache
def _real_word_list() -> tuple[str, ...]:
    """The subset of pyspellchecker's wordlist that also has a
    WordNet synset and meets minimum length. The teacher draws from
    this so its emissions match what the verifier considers real."""
    _ensure_wordnet()
    return tuple(
        w
        for w in _checker().word_frequency.dictionary
        if len(w) >= MIN_WORD_LEN and w.isalpha() and wordnet.synsets(w)
    )


def is_word(emission: str | bytes) -> bool:
    if isinstance(emission, bytes):
        try:
            emission = emission.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if not emission:
        return False
    word = emission.lower()
    if len(word) < MIN_WORD_LEN or not word.isalpha():
        return False
    _ensure_wordnet()
    return bool(wordnet.synsets(word))


def random_word() -> str:
    return random.choice(_real_word_list())

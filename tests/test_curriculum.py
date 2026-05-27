"""Graded adaptive curriculum with overlapping levels and per-level
thresholds. The teacher draws from focus +/- 1 each turn, and each
level has its own mastery bar — high for easy levels (must be
reliable), low for hard ones (a foothold counts)."""

from babbling_baby.curriculum import Curriculum


def test_curriculum_starts_at_focus_zero():
    c = Curriculum()
    assert c.focus == 0


def test_active_levels_at_bottom_includes_focus_and_next():
    c = Curriculum()
    assert c.active_levels() == (0, 1)


def test_active_levels_in_middle_includes_prev_focus_next():
    c = Curriculum()
    c.focus = 2
    assert c.active_levels() == (1, 2, 3)


def test_active_levels_at_top_includes_prev_and_focus():
    c = Curriculum()
    c.focus = len(c.LEVELS) - 1
    assert c.active_levels() == (c.focus - 1, c.focus)


def test_random_word_is_drawn_from_an_active_level():
    c = Curriculum()
    for _ in range(50):
        word = c.random_word()
        level, scored = c.score(word)
        assert scored
        assert level in c.active_levels()


def test_score_returns_level_and_hit():
    c = Curriculum()
    level, scored = c.score("a")
    assert scored
    assert level == 0


def test_score_returns_none_for_unknown_emission():
    c = Curriculum()
    level, scored = c.score("xyzq")
    assert not scored
    assert level is None


def test_unlock_thresholds_decrease_with_difficulty():
    """Easier levels demand higher mastery; harder ones accept a
    smaller foothold."""
    c = Curriculum()
    for i in range(len(c.UNLOCK_THRESHOLDS) - 1):
        assert c.UNLOCK_THRESHOLDS[i] >= c.UNLOCK_THRESHOLDS[i + 1]


def test_focus_advances_when_focus_window_meets_its_threshold():
    """At focus 0, threshold is 0.85. Feed 95% hits at level 0 — that
    should clear 0.85 and advance the focus."""
    c = Curriculum()
    initial_focus = c.focus
    threshold = c.UNLOCK_THRESHOLDS[c.focus]
    n_hits = int(c.PROMOTION_WINDOW * (threshold + 0.10))
    advanced = False
    for i in range(c.PROMOTION_WINDOW):
        emission = "a" if i < n_hits else "xyzqzq"
        if c.record(emission):
            advanced = True
    assert advanced
    assert c.focus == initial_focus + 1


def test_focus_does_not_advance_below_threshold():
    """At focus 0, a 50% hit rate is well under the 0.85 threshold —
    must not advance."""
    c = Curriculum()
    initial_focus = c.focus
    for i in range(c.PROMOTION_WINDOW * 2):
        emission = "a" if i % 2 == 0 else "xyzqzq"
        c.record(emission)
    assert c.focus == initial_focus


def test_promotion_clears_new_focus_window():
    """After promotion the new focus needs to earn its own track
    record — the prior level's success doesn't carry over."""
    c = Curriculum()
    threshold = c.UNLOCK_THRESHOLDS[0]
    for _ in range(c.PROMOTION_WINDOW):
        c.record("a")
    # Should now be at focus 1, and focus 1's window should be empty.
    assert c.focus == 1
    assert c.success_rate(1) == 0.0


def test_top_level_does_not_advance_further():
    c = Curriculum()
    c.focus = len(c.LEVELS) - 1
    last = c.focus
    for _ in range(c.PROMOTION_WINDOW * 2):
        c.record("hello")  # a level-something hit (varies by config)
    assert c.focus == last


def test_each_level_has_nonempty_vocab():
    c = Curriculum()
    for level in range(len(c.LEVELS)):
        assert len(c._vocab_lists[level]) > 0


def test_score_accepts_bytes_and_str():
    c = Curriculum()
    assert c.score(b"a") == c.score("a")


def test_random_word_can_draw_from_next_level_for_stretch():
    """At focus 0, the next level (1, two-letter words) should appear
    in random_word output some of the time as a stretch."""
    c = Curriculum()
    seen_levels = set()
    for _ in range(200):
        word = c.random_word()
        level, _ = c.score(word)
        if level is not None:
            seen_levels.add(level)
    # Both focus and next should show up over enough samples.
    assert 0 in seen_levels
    assert 1 in seen_levels

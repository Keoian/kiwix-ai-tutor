"""Unit tests for tutor.app.repetition_guard (host-side loop guard).

See docs/citation_experiment.md, "Seed exchange A/B (2026-09-20)", for the
runaway-answer defect this guards against: unbounded generation looping
on the same table row / sentence.
"""

from __future__ import annotations

from tutor.app.repetition_guard import find_repetition_loop


def test_no_loop_in_a_normal_short_answer():
    text = "Water boils at 100 degrees Celsius at sea level. [S1]"
    assert find_repetition_loop(text) is None


def test_sentence_repeated_four_times_is_detected():
    sentence = "Helium is the second lightest element in the periodic table."
    text = " ".join([sentence] * 4)
    loop = find_repetition_loop(text)
    assert loop is not None
    assert loop.repeat_count == 4
    # Trimmed text keeps exactly the first occurrence, drops the redundant repeats.
    assert loop.trimmed_text.strip() == sentence
    assert loop.trimmed_text.count(sentence) == 1


def test_sentence_repeated_three_times_is_not_yet_a_loop():
    sentence = "Helium is the second lightest element in the periodic table."
    text = " ".join([sentence] * 3)
    assert find_repetition_loop(text) is None


def test_table_row_repeated_on_newlines_is_detected():
    row = "| World War I | World War II | Cause |"
    text = "\n".join([row] * 5)
    loop = find_repetition_loop(text)
    assert loop is not None
    assert loop.repeat_count == 5
    assert loop.trimmed_text.strip() == row


def test_short_repeated_list_items_are_not_flagged():
    # A legitimate true/false table with many short rows must never trip
    # the guard -- this is exactly the "- yes" case the min-unit-length
    # threshold exists for.
    text = "\n".join(["- yes"] * 20)
    assert find_repetition_loop(text) is None


def test_case_and_whitespace_normalised_before_comparing():
    text = (
        "Photosynthesis makes sugar.\n  Photosynthesis makes sugar.  \n"
        "PHOTOSYNTHESIS MAKES SUGAR.\nphotosynthesis   makes sugar."
    )
    loop = find_repetition_loop(text)
    assert loop is not None
    assert loop.repeat_count == 4


def test_trailing_partial_repeat_still_counts_current_run_only():
    sentence = "The mitochondria is the powerhouse of the cell."
    other = "This is a different, unrelated sentence about biology class."
    text = " ".join([other, sentence, sentence, sentence, sentence])
    loop = find_repetition_loop(text)
    assert loop is not None
    assert loop.repeat_count == 4
    assert loop.trimmed_text.strip() == f"{other} {sentence}"


def test_empty_text_returns_none():
    assert find_repetition_loop("") is None


def test_custom_thresholds_are_respected():
    text = "\n".join(["ab"] * 10)
    # Below default min_unit_chars, so default call finds nothing...
    assert find_repetition_loop(text) is None
    # ...but a caller-supplied lower threshold does detect it.
    loop = find_repetition_loop(text, min_unit_chars=1, min_repeats=3)
    assert loop is not None
    assert loop.repeat_count == 10

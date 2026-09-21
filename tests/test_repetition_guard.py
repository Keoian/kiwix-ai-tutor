"""Unit tests for tutor.app.repetition_guard (host-side loop guard).

See docs/citation_experiment.md, "Seed exchange A/B (2026-09-20)", for the
runaway-answer defect this guards against: unbounded generation looping
on the same table row / sentence.
"""

from __future__ import annotations

from pathlib import Path

from tutor.app.repetition_guard import find_repetition_loop

_FIXTURES = Path(__file__).parent / "fixtures"


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


def test_real_loop_turn31_is_detected_and_trimmed_at_first_period():
    text = (_FIXTURES / "loop_turn31.txt").read_text(encoding="utf-8")
    loop = find_repetition_loop(text)
    assert loop is not None
    assert len(loop.trimmed_text) == 2363
    assert text.startswith(loop.trimmed_text)


def test_real_loop_turn32_is_detected_and_trimmed_at_first_period():
    text = (_FIXTURES / "loop_turn32.txt").read_text(encoding="utf-8")
    loop = find_repetition_loop(text)
    assert loop is not None
    assert len(loop.trimmed_text) == 2924
    assert text.startswith(loop.trimmed_text)


def test_real_loop_turn33_is_detected_and_trimmed_at_first_period():
    text = (_FIXTURES / "loop_turn33.txt").read_text(encoding="utf-8")
    loop = find_repetition_loop(text)
    assert loop is not None
    assert len(loop.trimmed_text) == 1554
    assert text.startswith(loop.trimmed_text)


def test_period_cycle_of_distinct_labelled_sentences_is_detected():
    # Mirrors the real defect: N distinct sentences cycling, each carrying
    # its own ever-incrementing invented [S#] label, repeated >= 3 times.
    cycle = [
        "This is the first cycled idea about fractions [S{}].",
        "This is the second cycled idea about fractions [S{}].",
        "This is the third cycled idea about fractions [S{}].",
    ]
    units = []
    label = 1
    for _ in range(4):
        for template in cycle:
            units.append(template.format(label))
            label += 1
    text = " ".join(units)
    loop = find_repetition_loop(text)
    assert loop is not None


def test_times_table_style_answer_is_not_flagged():
    # Distinct results each line -- must NOT trip, even though the guard
    # now tolerates near-identical units (decided: numeric content, unlike
    # an incrementing label, is the substance of the answer here, and
    # rows are short enough to fall under the min-unit-length rule anyway).
    text = "\n".join(f"3 x {i} = {3 * i}" for i in range(1, 11))
    assert find_repetition_loop(text) is None


def test_legit_numbered_list_of_eight_distinct_steps_is_not_flagged():
    text = "\n".join(
        f"Step {i}: perform distinct action number {i} in the sequence carefully."
        for i in range(1, 9)
    )
    assert find_repetition_loop(text) is None

"""RED/pinning tests for tutor.app.computed_check (docs/calc_investigation.md
fix #1, docs/attribution_design.md, spec v0.3 §14 "computed" statements).

Fast tests use a stub ``evaluate`` matching ``tutor.tools.calc_tool.evaluate``'s
contract (``{"ok": True, "result": "..."}`` / ``{"ok": False, "error": ...}``)
so the suite doesn't spawn a subprocess per assertion; a handful of
integration tests use the real sandboxed evaluator to prove the wiring
works end to end. No live LLM calls anywhere in this file.
"""

from __future__ import annotations

import json

import pytest

from tutor.app.computed_check import check_computed_statements
from tutor.tools.calc_tool import evaluate as real_evaluate


def _stub_evaluate(expr: str) -> dict:
    """A tiny, deterministic stand-in for the sandboxed evaluator: safe
    because test expressions here are hand-picked literals, never
    model-supplied text passed to eval()."""
    try:
        # Only +,-,*,/,** and parens ever appear in our normalized
        # expressions; still never used on untrusted input in production.
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "result": repr(float(value))}


def _one(items, status=None):
    if status is not None:
        items = [i for i in items if i["status"] == status]
    assert len(items) == 1, items
    return items[0]


# ---------------------------------------------------------------------------
# Fixture: the real 108.8-vs-80 error (data/granite_soak10_v3.turns.json
# index 32, "What is 12.5% of 640?")
# ---------------------------------------------------------------------------


def test_detects_the_108_8_vs_80_mismatch():
    answer = (
        "12.5% of 640 is calculated by multiplying 640 by 0.17 (since 17% is "
        "17/100 or 0.17) to get 108.8 [S1]."
    )
    items = check_computed_statements(answer, evaluate=_stub_evaluate)
    item = _one(items, status="mismatch")
    assert item["stated"] == pytest.approx(108.8)
    assert item["computed"] == pytest.approx(80.0)


def test_verifies_a_correct_percent_statement():
    answer = "12.5% of 640 is calculated by multiplying, which equals 80."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["computed"] == pytest.approx(80.0)
    assert item["stated"] == pytest.approx(80.0)


def test_helium_temperature_conversion_verified():
    answer = "Helium boils at **-268.928 C**, which is equivalent to **-452.070 F**."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["stated"] == pytest.approx(-452.070, abs=1e-2)
    assert item["computed"] == pytest.approx(-452.070, abs=1e-2)


def test_helium_temperature_conversion_mismatch():
    answer = "Helium boils at **-268.928 C**, which is equivalent to **-400.0 F**."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="mismatch")
    assert item["stated"] == pytest.approx(-400.0)


def test_celsius_kelvin_conversion():
    answer = "Water freezes at 0 C, which is 273.15 K."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["computed"] == pytest.approx(273.15)


# ---------------------------------------------------------------------------
# Chain arithmetic
# ---------------------------------------------------------------------------


def test_chain_multiplication_verified():
    answer = "2 x 3 x 4 = 24."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["computed"] == pytest.approx(24.0)


def test_chain_multiplication_with_unicode_times_mismatch():
    answer = "2 × 3 × 4 = 20."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="mismatch")
    assert item["computed"] == pytest.approx(24.0)


def test_thousands_separators_and_decimals():
    answer = "1,250.5 + 749.5 = 2,000."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["computed"] == pytest.approx(2000.0)


def test_unicode_minus_subtraction():
    answer = "10 − 3 = 7."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["computed"] == pytest.approx(7.0)


def test_division_and_caret_power():
    answer = "10 ÷ 2 = 5 and 2 ^ 3 = 8."
    items = check_computed_statements(answer, evaluate=_stub_evaluate)
    verified = [i for i in items if i["status"] == "verified"]
    assert len(verified) == 2


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------


def test_rounding_tolerance_accepts_rounded_stated_value():
    answer = "2/3 + 1/6 = 0.83."  # true value 0.8333...
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["computed"] == pytest.approx(0.8333333333, abs=1e-6)


def test_relative_tolerance_accepts_tiny_float_noise():
    answer = "0.1 + 0.2 = 0.3."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["computed"] == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Question-side gap: the answer never states the computed number at all.
# ---------------------------------------------------------------------------


def test_question_gap_flags_missing_answer_number():
    question = "What is 12.5% of 640?"
    answer = "That's a great question about percentages! Let's think about it."
    item = _one(check_computed_statements(answer, question, evaluate=_stub_evaluate))
    assert item["span"] is None
    assert item["status"] == "mismatch"
    assert item["computed"] == pytest.approx(80.0)


def test_question_gap_not_flagged_when_answer_has_the_number():
    question = "What is 12.5% of 640?"
    answer = "12.5% of 640 is 80."
    items = check_computed_statements(answer, question, evaluate=_stub_evaluate)
    assert all(i["span"] is not None for i in items)


# ---------------------------------------------------------------------------
# Negatives: conservative, ambiguous forms are never reported.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "The war lasted from 1939-1945.",
        "See pages 3-4 for details.",
        "The ratio of flour to sugar is 3:4.",
        "As shown in [S12], the boiling point is notable.",
        "1. First step. 2. Second step. 3. Third step.",
        "Water is H2O.",
        "Solve for x: x + 2 = 5.",
        "2x = 10, so x is a variable here.",
        "CO2 and H2O are common molecules.",
    ],
)
def test_negatives_emit_nothing(answer):
    assert check_computed_statements(answer, evaluate=_stub_evaluate) == []


def test_negative_year_with_equals_like_words_not_confused():
    # "is" here is not an arithmetic result marker for a bare year.
    answer = "The year 1969 is when Apollo 11 landed."
    assert check_computed_statements(answer, evaluate=_stub_evaluate) == []


# ---------------------------------------------------------------------------
# Real sandboxed evaluator (integration, no live LLM): proves the wiring
# from computed_check into tutor.tools.calc_tool.evaluate actually works.
# ---------------------------------------------------------------------------


def test_real_evaluator_percent_mismatch():
    answer = "12.5% of 640 is 108.8."
    item = _one(check_computed_statements(answer, evaluate=real_evaluate), status="mismatch")
    assert item["computed"] == pytest.approx(80.0)


def test_real_evaluator_chain_verified():
    answer = "2 x 3 x 4 = 24."
    item = _one(check_computed_statements(answer, evaluate=real_evaluate), status="verified")
    assert item["computed"] == pytest.approx(24.0)


def test_real_evaluator_helium():
    answer = "Helium boils at **-268.928 C**, which is equivalent to **-452.070 F**."
    item = _one(check_computed_statements(answer, evaluate=real_evaluate), status="verified")
    assert item["computed"] == pytest.approx(-452.070, abs=1e-2)


# ---------------------------------------------------------------------------
# Never crashes on a broken/failing evaluator.
# ---------------------------------------------------------------------------


def test_failing_evaluator_drops_the_item_instead_of_raising():
    def _always_fails(expr: str) -> dict:
        return {"ok": False, "error": "nope"}

    answer = "2 x 3 x 4 = 24."
    assert check_computed_statements(answer, evaluate=_always_fails) == []


def test_evaluator_returning_garbage_is_tolerated():
    def _garbage(expr: str) -> dict:
        return {"ok": True, "result": "not-a-number"}

    answer = "2 x 3 x 4 = 24."
    assert check_computed_statements(answer, evaluate=_garbage) == []


# ---------------------------------------------------------------------------
# Real soak data fixtures (data/*.turns.json) -- counts reported by the
# calling agent, not asserted rigidly here since the exact answers are
# recorded transcripts, but at least the known error must be caught.
# ---------------------------------------------------------------------------


def _load_answer_by_question(path, question_text):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for turn in data:
        if turn.get("input_text") == question_text:
            return turn.get("input_text"), turn.get("answer_text")
    raise AssertionError(f"{question_text!r} not found in {path}")


def test_soak_v3_index_32_is_the_108_8_error():
    question, answer = _load_answer_by_question(
        "data/granite_soak10_v3.turns.json", "What is 12.5% of 640?"
    )
    items = [
        i
        for i in check_computed_statements(answer, question, evaluate=_stub_evaluate)
        if i["status"] == "mismatch"
    ]
    assert items, "expected at least one mismatch item for the 108.8-vs-80 turn"
    assert any(
        i["stated"] == pytest.approx(108.8) and i["computed"] == pytest.approx(80.0)
        for i in items
    )

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
from pathlib import Path

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
# Real soak answers, verbatim -- but from a small COMMITTED fixture, never
# from data/*.turns.json: data/ is gitignored, so a test that opens it
# passes locally (where a real soak run has been done) and raises
# FileNotFoundError on a clean checkout / CI (windows + ubuntu). See
# tests/fixtures/computed_check_answers.json (copied verbatim from
# data/granite_soak10_v2.turns.json / data/granite_soak10_v3.turns.json;
# the full soak fixture tally itself is a manual/eval-time check, not a
# suite test -- see the handback report for those counts).
# ---------------------------------------------------------------------------

_FIXTURES_FILE = Path(__file__).resolve().parent / "fixtures" / "computed_check_answers.json"
_FIXTURE_ANSWERS = json.loads(_FIXTURES_FILE.read_text(encoding="utf-8"))


def _load_answer_by_question(soak_name, question_text):
    return question_text, _FIXTURE_ANSWERS[soak_name][question_text]


# ---------------------------------------------------------------------------
# 2026-09-20 follow-up: the loose "last number in the sentence" heuristic
# produced FALSE mismatches when the answer states the arithmetic in one
# sentence ("...by 0.125 (since 12.5% is 12.5/100 or 0.125).") and the
# actual claimed result in the NEXT sentence ("So, 640 x 0.125 = 80.").
# Fix: verified if ANY number anywhere in the answer matches the computed
# value; mismatch only when a number is tightly bound to the expression by
# is/=/equals/gives/comes to (not "is calculated by", "you multiply",
# "means") AND no number anywhere in the answer matches. Real full answers
# from data/granite_soak10_v2.turns.json (graded correct by the soak).
# ---------------------------------------------------------------------------


def test_no_false_mismatch_when_correct_value_is_in_a_later_sentence_12_5_percent():
    question, answer = _load_answer_by_question(
        "granite_soak10_v2", "What is 12.5% of 640?"
    )
    items = check_computed_statements(answer, question, evaluate=_stub_evaluate)
    mismatches = [i for i in items if i["status"] == "mismatch"]
    assert mismatches == [], mismatches
    assert any(i["status"] == "verified" and i["computed"] == pytest.approx(80.0) for i in items)


def test_no_false_mismatch_when_correct_value_is_in_a_later_sentence_250_percent():
    question, answer = _load_answer_by_question(
        "granite_soak10_v2", "What is 250% of 40?"
    )
    items = check_computed_statements(answer, question, evaluate=_stub_evaluate)
    mismatches = [i for i in items if i["status"] == "mismatch"]
    assert mismatches == [], mismatches
    assert any(i["status"] == "verified" and i["computed"] == pytest.approx(100.0) for i in items)


def test_tightly_bound_wrong_result_is_still_a_mismatch():
    answer = "17% of 240 = 999."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="mismatch")
    assert item["stated"] == pytest.approx(999.0)
    assert item["computed"] == pytest.approx(40.8)


def test_loosely_worded_claim_with_no_tight_binding_and_no_match_emits_nothing():
    # No connector ever directly binds a number to the expression, and the
    # correct value never appears anywhere -- conservative: say nothing
    # rather than guess which number (if any) was meant as the answer.
    answer = "17% of 240 involves multiplying by 0.17, roughly speaking."
    assert check_computed_statements(answer, evaluate=_stub_evaluate) == []


def test_temperature_mismatch_suppressed_when_correct_value_appears_elsewhere():
    answer = (
        "Helium boils at -268.928 C, which is roughly -400.0 F. "
        "(Corrected: -268.928 C is actually -452.070 F.)"
    )
    items = check_computed_statements(answer, evaluate=_stub_evaluate)
    assert all(i["status"] != "mismatch" for i in items), items


def test_soak_v3_index_32_is_the_108_8_error():
    question, answer = _load_answer_by_question(
        "granite_soak10_v3", "What is 12.5% of 640?"
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


# ---------------------------------------------------------------------------
# Dedupe: the answer-scan and the question-gap check must never both flag
# the SAME underlying error -- one real mistake, one item.
# ---------------------------------------------------------------------------


def test_question_gap_check_does_not_duplicate_an_answer_scan_mismatch():
    question = "What is 12.5% of 640?"
    answer = "12.5% of 640 is calculated by multiplying 640 by 0.17 to get 108.8."
    items = [
        i
        for i in check_computed_statements(answer, question, evaluate=_stub_evaluate)
        if i["status"] == "mismatch" and i["computed"] == pytest.approx(80.0)
    ]
    assert len(items) == 1, items


def test_question_gap_check_still_fires_when_answer_scan_found_nothing():
    question = "What is 12.5% of 640?"
    answer = "That's a great question about percentages! Let's think about it."
    items = check_computed_statements(answer, question, evaluate=_stub_evaluate)
    mismatches = [i for i in items if i["status"] == "mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0]["span"] is None
    assert mismatches[0]["computed"] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# 2026-09-21 "✓ checked" wording bug (owner-reported live bug B): a bare
# "✓ checked" reads as "this fact was checked" when the host only re-did a
# unit conversion arithmetic step, never the underlying claim (the owner
# saw "-70C (-94F checked)" for a fabricated temperature). Every item now
# carries its "kind" so the UI can special-case unit conversions.
# ---------------------------------------------------------------------------


def test_temperature_conversion_item_is_tagged_kind_temp():
    answer = "Helium boils at **-268.928 C**, which is equivalent to **-452.070 F**."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["kind"] == "temp"


def test_chain_arithmetic_item_is_tagged_kind_chain():
    answer = "2 x 3 x 4 = 24"
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["kind"] == "chain"


def test_percent_item_is_tagged_kind_percent():
    answer = "12.5% of 640 is calculated by multiplying, which equals 80."
    item = _one(check_computed_statements(answer, evaluate=_stub_evaluate), status="verified")
    assert item["kind"] == "percent"


# ---------------------------------------------------------------------------
# 2026-09-21 owner-reported LIVE crash (Ling, "What is puberty?"): a Unicode
# minus sign (U+2212, "−") or en dash used as a minus ("–") in the model's
# text must never raise out of check_computed_statements and fail the
# turn. Also: check_computed_statements itself must be exception-proof --
# a checker crashing must never break an answered turn.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "It got as low as −70°C (−94°F).",
        "The pH dropped by –5 over the week.",
        "10−3 is seven.",
        "Kids usually grow 1–2 years apart.",
        "A healthy blood pH is 7.4–7.8.",
    ],
)
def test_unicode_minus_and_en_dash_inputs_never_raise(answer):
    # Real evaluator (subprocess), not the stub -- exercises the actual
    # production path end to end. Must not raise, whatever it returns.
    check_computed_statements(answer)


def test_unicode_minus_temperature_still_verifies_correctly():
    answer = "It got as low as −70°C (−94°F)."
    items = check_computed_statements(answer, evaluate=_stub_evaluate)
    item = _one(items, status="verified")
    assert item["kind"] == "temp"
    assert item["computed"] == pytest.approx(-94.0, abs=1e-2)


def test_unicode_minus_temperature_mismatch_does_not_raise():
    # Regression for the exact live crash: a MISMATCHING temperature pair
    # (forcing _within_tolerance's decimal-rounding fallback branch) whose
    # stated value is written with a Unicode minus sign must not raise.
    answer = "It got as low as −70°C (−90°F)."
    items = check_computed_statements(answer, evaluate=_stub_evaluate)
    item = _one(items, status="mismatch")
    assert item["stated"] == pytest.approx(-90.0)


def test_digit_en_dash_digit_range_produces_no_chain_item():
    # "1–2 years" is a range, not a computation -- must not be
    # misread as chain arithmetic.
    answer = "Kids usually grow 1–2 years apart."
    items = check_computed_statements(answer, evaluate=_stub_evaluate)
    assert items == []


def test_check_computed_statements_never_propagates_a_raising_evaluator():
    """A checker must never break an answered turn: if the evaluator
    itself raises (or anything else inside the scan blows up), the
    function swallows it and returns "no computed items" rather than
    propagating."""

    def _boom(expr: str) -> dict:
        raise RuntimeError("evaluator exploded")

    answer = "12.5% of 640 is 80."
    assert check_computed_statements(answer, evaluate=_boom) == []

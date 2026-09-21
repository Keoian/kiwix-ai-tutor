"""Host-side verification of stated arithmetic ("computed" statements).

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §14 (the tutor's
three kinds of statement: source-backed, **computed**, own-example) and
docs/calc_investigation.md fix #1 (Granite skips the offered `calc` tool
inside lessons and does arithmetic in its head; measured error: "12.5% of
640" answered 108.8, true value 80).

This module never calls ``eval()``/``exec()`` -- every candidate expression
found in the model's own text is normalized to the small Python-arithmetic
subset :mod:`tutor.tools.calc_tool` accepts and handed to its sandboxed
``evaluate`` (subprocess, ast-whitelisted). It only ever *reads* the
model's answer; it never edits, rewrites, or removes anything.

Detected forms (conservative: an ambiguous match is dropped rather than
guessed at):

- A chain of numbers and operators ending in an explicit ``=``, e.g.
  ``"2 x 3 x 4 = 24"`` (also ``×``, ``*``, ``+``, ``-``/``−``, ``÷``/``/``,
  ``^``, thousands separators, decimals, parentheses).
- ``"p% of n is/are/equals/= c"``.
- A temperature stated in both Celsius and Fahrenheit or Kelvin in the
  same stretch of text (``"-268.928 C ... -452.070 F"``).

Deliberately NOT matched (left for the model, never guessed): bare ranges
("pages 3-4"), ratios ("3:4"), years, ``[S#]`` citation-label digits,
leading list numbering ("1. ", "2) "), chemical formulas ("H2O"), and
equations that contain a variable ("x + 2 = 5") -- all excluded by
requiring every operand to be a plain, word-boundary-delimited number, and
requiring an explicit ``=``/``is``/``are``/``equals`` result marker.

Also checks the student's own *question*: if it contains one of the
detectable computable forms (currently: "p% of n") and the answer contains
no number equal to the computed result (within tolerance), a "mismatch"
item is emitted with ``span=None``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from tutor.tools.calc_tool import evaluate as _default_evaluate

# ---------------------------------------------------------------------------
# Shared number/operator vocabulary
# ---------------------------------------------------------------------------

_NUM = r"[-−]?\d[\d,]*(?:\.\d+)?"
_OP_SYM = r"[+\-−×÷*/^]"
# "x"/"X" only counts as a multiplication operator when set off by
# whitespace on both sides ("2 x 3"), never adjoining a number ("2x", an
# algebraic term) or standing alone as a variable ("x + 2 = 5").
_OP = rf"(?:{_OP_SYM}|(?<=\s)[xX](?=\s))"

# A bare number, guarded against being part of a longer alphanumeric token
# (chemical formulas like "H2O", citation-label digits like "[S12]", a
# unit suffix glued on with no space).
_GUARDED_NUM = rf"(?<![A-Za-z0-9])({_NUM})(?![A-Za-z0-9])"

_CHAIN_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_NUM}(?:\s*{_OP}\s*{_NUM})+)\s*=\s*({_NUM})(?![A-Za-z0-9])"
)

_PERCENT_BARE_RE = re.compile(
    rf"{_GUARDED_NUM}\s*%\s*of\s*{_GUARDED_NUM}",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(text)))
    return [(s, e) for s, e in bounds if s < e]


_TEMP_C_RE = re.compile(rf"{_GUARDED_NUM}\s*(?:\xb0|degrees?)?\s*C\b")
_TEMP_F_RE = re.compile(rf"{_GUARDED_NUM}\s*(?:\xb0|degrees?)?\s*F\b")
_TEMP_K_RE = re.compile(rf"{_GUARDED_NUM}\s*(?:\xb0|degrees?)?\s*K\b")

# How far apart (characters) two temperature figures may be and still be
# read as "the same statement, restated in another unit".
_TEMP_PAIR_WINDOW = 80

_REL_TOL = 1e-3

EvaluateFn = Callable[[str], dict]


def _to_float(num_str: str) -> float:
    return float(num_str.replace(",", "").replace("−", "-"))


def _normalize_expr(expr: str) -> str:
    """Turn a matched, human-written expression into the small Python
    arithmetic subset ``tutor.tools.calc_tool.evaluate`` accepts."""
    expr = expr.replace(",", "")
    expr = expr.replace("−", "-")
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = expr.replace("^", "**")
    expr = re.sub(r"(?<=\s)[xX](?=\s)", "*", expr)
    return expr


def _within_tolerance(stated: float, computed: float, stated_str: str) -> bool:
    if computed == stated:
        return True
    denom = max(abs(computed), abs(stated), 1e-9)
    if abs(computed - stated) / denom <= _REL_TOL:
        return True
    decimals = 0
    if "." in stated_str:
        decimals = len(stated_str.split(".")[-1])
    return round(computed, decimals) == round(float(stated_str.replace(",", "")), decimals)


def _run_evaluate(evaluate: EvaluateFn, expr: str) -> float | None:
    try:
        result = evaluate(expr)
    except Exception:  # noqa: BLE001 - never let a bad expression crash the check
        return None
    if not isinstance(result, dict) or not result.get("ok"):
        return None
    try:
        return float(result.get("result", ""))
    except (TypeError, ValueError):
        return None


def _chain_items(text: str, evaluate: EvaluateFn) -> list[dict]:
    items = []
    for m in _CHAIN_RE.finditer(text):
        raw_expr, stated_str = m.group(1), m.group(2)
        computed = _run_evaluate(evaluate, _normalize_expr(raw_expr))
        if computed is None:
            continue
        stated = _to_float(stated_str)
        status = "verified" if _within_tolerance(stated, computed, stated_str) else "mismatch"
        items.append(
            {
                "span": (m.start(1), m.end(2)),
                "expression": f"{raw_expr.strip()} = {stated_str}",
                "stated": stated,
                "computed": computed,
                "status": status,
            }
        )
    return items


# A number introduced with one of these connectors, directly after "p% of
# n" or later in the same sentence, is being offered as THE result of the
# computation -- as opposed to an intermediate value mentioned in passing
# ("is calculated by", "you multiply", "means", "or" as in "12.5/100 or
# 0.125"). "to get"/"get"/"gives"/"comes to" all count, since "... to get
# 108.8" is the real, measured shape of Granite's error
# (docs/calc_investigation.md fix #1).
_TIGHT_RESULT_RE = re.compile(
    rf"(?:is|=|equals|gives|comes to|to get|get)\s*{_GUARDED_NUM}",
    re.IGNORECASE,
)


def _answer_matching_number_span(text: str, value: float) -> tuple[int, int] | None:
    """The span of the first sentence in ``text`` containing a plain
    number equal to ``value`` (tolerance), or ``None``."""
    for sent_start, sent_end in _sentence_spans(text):
        sentence = text[sent_start:sent_end]
        for m in re.finditer(_GUARDED_NUM, sentence):
            try:
                candidate = _to_float(m.group(1))
            except ValueError:
                continue
            if _within_tolerance(candidate, value, m.group(1)):
                return (sent_start, sent_end)
    return None


def _tight_bound_result(sentence: str, after: int) -> re.Match | None:
    """The LAST connector-bound number in ``sentence`` at or after offset
    ``after``, skipping a match that is really "N% is M" restating the
    percent as a fraction (the text right before the connector word ends
    in ``%``) rather than stating a final result."""
    best = None
    for m in _TIGHT_RESULT_RE.finditer(sentence, after):
        prefix = sentence[: m.start()].rstrip()
        if prefix.endswith("%"):
            continue
        best = m
    return best


def _percent_items(text: str, evaluate: EvaluateFn) -> list[dict]:
    """Find "p% of n" claims. Status is "verified" if ANY number anywhere
    in the whole answer equals the computed value (covers real prose that
    states the arithmetic in one sentence and the result in the next,
    e.g. "...by 0.125 (since 12.5% is 12.5/100 or 0.125). So, 640 x 0.125
    = 80." -- graded correct, previously a false "mismatch" from a
    sentence-local last-number heuristic). Otherwise, "mismatch" only when
    a number is tightly bound to the expression by a result connector
    (see ``_TIGHT_RESULT_RE``) and disagrees. Anything else (a loosely
    worded claim with no tight binding and no matching number anywhere)
    emits nothing -- conservative, no guessing."""
    items = []
    for sent_start, sent_end in _sentence_spans(text):
        sentence = text[sent_start:sent_end]
        m = _PERCENT_BARE_RE.search(sentence)
        if not m:
            continue
        p_str, n_str = m.group(1), m.group(2)
        expr = f"({_normalize_expr(p_str)}/100)*{_normalize_expr(n_str)}"
        computed = _run_evaluate(evaluate, expr)
        if computed is None:
            continue

        match_span = _answer_matching_number_span(text, computed)
        if match_span is not None:
            items.append(
                {
                    "span": match_span,
                    "expression": f"{p_str}% of {n_str}",
                    "stated": computed,
                    "computed": computed,
                    "status": "verified",
                }
            )
            continue

        tight = _tight_bound_result(sentence, m.end())
        if tight is None:
            continue
        c_str = tight.group(1)
        stated = _to_float(c_str)
        items.append(
            {
                "span": (sent_start + m.start(), sent_end),
                "expression": f"{p_str}% of {n_str} = {c_str}",
                "stated": stated,
                "computed": computed,
                "status": "mismatch",
            }
        )
    return items


def _temp_items(text: str, evaluate: EvaluateFn) -> list[dict]:
    items = []
    c_matches = list(_TEMP_C_RE.finditer(text))
    if not c_matches:
        return items
    other_matches = [(m, "F") for m in _TEMP_F_RE.finditer(text)] + [
        (m, "K") for m in _TEMP_K_RE.finditer(text)
    ]
    used_others: set[int] = set()
    for cm in c_matches:
        best = None
        best_dist = None
        for idx, (om, unit) in enumerate(other_matches):
            if idx in used_others:
                continue
            # A Fahrenheit/Kelvin figure whose own digits fell inside the
            # Celsius match (overlapping spans) is not a second value.
            if om.start() < cm.end() and om.end() > cm.start():
                continue
            dist = min(abs(om.start() - cm.end()), abs(cm.start() - om.end()))
            if dist > _TEMP_PAIR_WINDOW:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (idx, om, unit)
        if best is None:
            continue
        idx, om, unit = best
        used_others.add(idx)
        stated = _to_float(om.group(1))
        if unit == "F":
            expr = f"({_normalize_expr(cm.group(1))})*9/5+32"
        else:
            expr = f"({_normalize_expr(cm.group(1))})+273.15"
        computed = _run_evaluate(evaluate, expr)
        if computed is None:
            continue
        if _within_tolerance(stated, computed, om.group(1)):
            status = "verified"
        elif _answer_has_number_near(text, computed):
            # The right value is stated somewhere else in the answer (e.g.
            # a correction, or a different sentence) -- this particular
            # pairing disagrees, but a false "mismatch" on an otherwise
            # correct answer is worse than saying nothing about it.
            continue
        else:
            status = "mismatch"
        span = (min(cm.start(), om.start()), max(cm.end(), om.end()))
        items.append(
            {
                "span": span,
                "expression": f"{cm.group(1)} C -> {unit}",
                "stated": stated,
                "computed": computed,
                "status": status,
            }
        )
    return items


def _answer_has_number_near(text: str, value: float) -> bool:
    for m in re.finditer(_GUARDED_NUM, text):
        try:
            candidate = _to_float(m.group(1))
        except ValueError:
            continue
        if _within_tolerance(candidate, value, m.group(1)):
            return True
    return False


def _question_gap_items(
    question_text: str | None, answer_text: str, evaluate: EvaluateFn
) -> list[dict]:
    if not question_text:
        return []
    items = []
    for m in _PERCENT_BARE_RE.finditer(question_text):
        p_str, n_str = m.group(1), m.group(2)
        expr = f"({_normalize_expr(p_str)}/100)*{_normalize_expr(n_str)}"
        computed = _run_evaluate(evaluate, expr)
        if computed is None:
            continue
        if _answer_has_number_near(answer_text, computed):
            continue
        items.append(
            {
                "span": None,
                "expression": f"{p_str}% of {n_str}",
                "stated": None,
                "computed": computed,
                "status": "mismatch",
            }
        )
    return items


def check_computed_statements(
    answer_text: str,
    question_text: str | None = None,
    *,
    evaluate: EvaluateFn | None = None,
) -> list[dict]:
    """Scan ``answer_text`` (and, for the "the answer skipped the number
    entirely" case, ``question_text``) for explicit arithmetic claims and
    verify each against ``evaluate`` (defaults to
    ``tutor.tools.calc_tool.evaluate``). Returns a list of items,
    ``{"span", "expression", "stated", "computed", "status"}``, in the
    order found. Never raises for malformed input; a candidate expression
    the evaluator itself rejects is simply dropped, not reported."""
    evaluate = evaluate or _default_evaluate
    items: list[dict] = []
    items.extend(_percent_items(answer_text, evaluate))

    percent_spans = [it["span"] for it in items]

    def _overlaps_percent(span: tuple[int, int]) -> bool:
        return any(not (span[1] <= s or span[0] >= e) for (s, e) in percent_spans)

    for item in _chain_items(answer_text, evaluate):
        if _overlaps_percent(item["span"]):
            continue
        items.append(item)

    items.extend(_temp_items(answer_text, evaluate))
    items.extend(_question_gap_items(question_text, answer_text, evaluate))
    items.sort(key=lambda it: (it["span"] is None, it["span"] or (0, 0)))
    return items

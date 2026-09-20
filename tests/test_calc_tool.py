"""RED tests for tutor.tools.calc_tool.evaluate.

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §9.2 (calc
contract: `{ "ok": true, "result": "..." }` / `{ "ok": false, "error": "..." }`,
sympy whitelist, <=200 chars, subprocess with 1 s wall clock + 64 MB memory
cap) and docs/plan/offline_tutor_implementation_plan.md WP-C2 (Windows has no
`resource` module: subprocess timeout for wall clock + psutil polling for
memory; same interface so Linux can add RLIMIT_AS later).

Contract decisions made here (spec silent or ambiguous):
- Public signature is `evaluate(expression, *, timeout_s=1.0,
  memory_limit_mb=64) -> dict`, matching the spec's literal caps exactly
  (<=200 chars, 64 MB). The 64 MB is enforced as *growth above the
  child's own post-import baseline RSS* (see tutor/tools/calc_tool.py
  module docstring and docs/calc_tool.md): a fresh interpreter that has
  merely imported sympy can itself sit close to or above 64 MB RSS, so a
  literal absolute 64 MB ceiling measured from zero would reject even
  trivial expressions on some machines.
- `solve(<linear/quadratic equation in x>)` is accepted as a single
  extra expression form, e.g. `solve(2*x + 3 = 11)` -> `"x = 4"` or
  `solve(x**2 - 5*x + 6 = 0)` -> `"x = 2 or x = 3"`. This is documented
  in the tool description in tutor/tools/schemas.py.
- When an exact result is not already a plain integer/decimal (i.e. an
  irrational `solve()` root), the host appends a decimal approximation,
  e.g. `"x = sqrt(2) ≈ 1.41421356237"`.
- Deterministic formatting: exact rationals render as "a/b" (no
  decimal), integers render without a trailing ".0", floats are rendered
  to <= 12 significant digits.
- Division by zero, unknown names, disallowed calls, and syntax errors are
  ok:false with a readable message, never a raised exception.
- The huge-exponent case ("9**9**9") must return within the timeout as an
  ok:false error (or a bounded ok:true) rather than hang the parent.

Everything here runs the real subprocess sandbox (no fake LLM/research is
relevant to this module); network is never touched by calc.
"""

from __future__ import annotations

import time

import pytest

from tutor.tools.calc_tool import evaluate

KNOWN_ANSWERS: list[tuple[str, str]] = [
    ("2 + 2", "4"),
    ("4871*392", "1909432"),
    ("10 - 3", "7"),
    ("6 * 7", "42"),
    ("100 / 4", "25"),
    ("2 ** 10", "1024"),
    ("sqrt(16)", "4"),
    ("sqrt(2)", "1.4142135624"),
    ("1/3 + 1/6", "1/2"),
    ("1/2 + 1/2", "1"),
    ("3/4 * 2/3", "1/2"),
    ("2.5 + 2.5", "5"),
    ("0.1 + 0.2", "0.3"),
    ("-5 + 3", "-2"),
    ("(2 + 3) * 4", "20"),
    ("2 + 3 * 4", "14"),
    ("pi", "3.1415926536"),
    ("E", "2.7182818285"),
    ("log(E)", "1"),
    ("sin(0)", "0"),
    ("cos(0)", "1"),
    ("Rational(1, 3)", "1/3"),
    ("7 % 3", "1"),
    ("abs(-9)", "9"),
    ("(1 + 2) ** 2 - 4", "5"),
]


@pytest.mark.parametrize("expression, expected", KNOWN_ANSWERS)
def test_known_answers(expression, expected):
    result = evaluate(expression)
    assert result["ok"] is True
    assert result["result"] == expected


def test_known_answer_table_has_at_least_25_entries():
    assert len(KNOWN_ANSWERS) >= 25


def test_result_shape_on_success():
    result = evaluate("2 + 2")
    assert set(result.keys()) == {"ok", "result"}
    assert result["ok"] is True
    assert isinstance(result["result"], str)


def test_result_shape_on_failure():
    result = evaluate("1/0")
    assert set(result.keys()) == {"ok", "error"}
    assert result["ok"] is False
    assert isinstance(result["error"], str)
    assert result["error"]


def test_division_by_zero_is_ok_false_readable():
    result = evaluate("1/0")
    assert result["ok"] is False
    assert "zero" in result["error"].lower() or "divi" in result["error"].lower()


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "().__class__",
        "().__class__.__bases__",
        "open('x')",
        "eval('1+1')",
        "exec('1+1')",
        "compile('1+1', '<s>', 'eval')",
        "'a string'",
        "lambda x: x",
        "[x for x in range(10)]",
        "x = 5",
        "import os",
        "os.system('dir')",
        "globals()",
        "locals()",
        "getattr(1, '__class__')",
        "__builtins__",
    ],
)
def test_rejects_dangerous_or_disallowed_expressions(expression):
    result = evaluate(expression)
    assert result["ok"] is False
    assert "error" in result


def test_rejects_expression_over_200_chars():
    expression = "1+" * 120 + "1"
    assert len(expression) > 200
    result = evaluate(expression)
    assert result["ok"] is False


def test_huge_exponent_does_not_hang_and_returns_within_timeout():
    start = time.monotonic()
    result = evaluate("9**9**9", timeout_s=1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 3.0
    assert isinstance(result, dict)
    assert "ok" in result


def test_artificially_slow_expression_is_killed_at_wall_clock_and_parent_survives():
    # A deliberately pathological, extremely slow-to-evaluate expression:
    # deeply nested factorial-like symbolic blowup that should not finish
    # inside the 1s budget.
    slow_expression = "factorial(factorial(7))"
    start = time.monotonic()
    result = evaluate(slow_expression, timeout_s=1.0)
    elapsed = time.monotonic() - start

    assert elapsed < 3.0
    assert result["ok"] is False

    # Parent process is alive and can keep evaluating.
    followup = evaluate("2 + 2")
    assert followup["ok"] is True
    assert followup["result"] == "4"


def test_syntax_error_is_ok_false_not_an_exception():
    result = evaluate("2 +* 2")
    assert result["ok"] is False


def test_empty_expression_is_ok_false():
    result = evaluate("")
    assert result["ok"] is False


def test_deterministic_integer_result_has_no_trailing_decimal():
    result = evaluate("6 * 7")
    assert result["ok"] is True
    assert result["result"] == "42"
    assert "." not in result["result"]


def test_float_result_limited_to_twelve_significant_digits():
    result = evaluate("sqrt(2)")
    assert result["ok"] is True
    digits = result["result"].replace(".", "").replace("-", "")
    assert len(digits) <= 12


def test_result_is_always_a_string_type():
    for expression in ["2+2", "1/3", "sqrt(2)", "pi"]:
        result = evaluate(expression)
        assert isinstance(result["result"], str)


def test_evaluate_runs_in_a_subprocess_not_the_parent_interpreter():
    # A dangerous expression that would corrupt the parent's own sys.modules
    # if it ran in-process must not affect this test process at all.
    import sys

    marker = "tutor_test_marker_module"
    assert marker not in sys.modules
    evaluate(f"__import__('sys').modules['{marker}'] = 1")
    assert marker not in sys.modules


# ---------------------------------------------------------------------------
# (a) basic solving of one linear/quadratic equation in x
# ---------------------------------------------------------------------------


def test_solve_linear_equation():
    result = evaluate("solve(2*x + 3 = 11)")
    assert result["ok"] is True
    assert result["result"] == "x = 4"


def test_solve_quadratic_equation_two_integer_roots():
    result = evaluate("solve(x**2 - 5*x + 6 = 0)")
    assert result["ok"] is True
    assert result["result"] == "x = 2 or x = 3"


def test_solve_rejects_more_than_one_equals_sign():
    result = evaluate("solve(x = 1 = 2)")
    assert result["ok"] is False


def test_solve_rejects_disallowed_symbol():
    result = evaluate("solve(2*y + 3 = 11)")
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# (b) decimal approximation appended when the exact result isn't already
# a plain integer/decimal (exercised via an irrational solve() root).
# ---------------------------------------------------------------------------


def test_solve_irrational_root_appends_decimal_approximation():
    result = evaluate("solve(x**2 - 2 = 0)")
    assert result["ok"] is True
    assert "sqrt(2)" in result["result"]
    assert "≈" in result["result"]
    assert "1.41421356237" in result["result"]

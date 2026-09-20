"""Child process entry point for the calc tool's sandboxed evaluation.

Invoked as ``python -m tutor.tools.calc_worker``. This module never trusts
model-supplied text enough to ``eval()``/``exec()`` it: every expression is
first parsed with :mod:`ast` and walked against an explicit whitelist of
node types, names, and function calls (see :func:`_ast_to_sympy`); only
after that whitelist walk succeeds are the corresponding sympy objects and
operators used to build the actual expression.

Protocol (line-delimited JSON over stdin/stdout, both UTF-8):

1. On startup, after importing sympy, this process writes one line
   ``{"ready": true, "baseline_rss": <bytes>}`` reporting its own RSS.
   The parent (tutor.tools.calc_tool.evaluate) uses this as the baseline
   for the growth-based 64 MB memory cap, since a freshly-imported sympy
   interpreter's own RSS varies by machine and can itself approach the
   cap (see docs/calc_tool.md for a measured baseline on this machine).
2. It then reads exactly one line, a JSON object ``{"expression": "..."}``.
3. It writes exactly one line, a JSON object ``{"ok": true, "result":
   "..."}`` or ``{"ok": false, "error": "..."}``, and exits.

A single extra expression form beyond plain arithmetic is accepted:
``solve(<linear/quadratic equation in x>)``, e.g. ``solve(2*x + 3 = 11)``
or ``solve(x**2 - 5*x + 6 = 0)``, with exactly one ``=`` and the single
symbol ``x``. When a root is not already a plain integer/rational, a
decimal approximation is appended, e.g. ``"x = sqrt(2) ≈
1.41421356237"``.
"""

from __future__ import annotations

import ast
import json
import re
import sys

_ALLOWED_FUNCS = {
    "sqrt",
    "sin",
    "cos",
    "tan",
    "log",
    "exp",
    "abs",
    "factorial",
    "Rational",
}
_SOLVE_RE = re.compile(r"^solve\((.*)\)$", re.DOTALL)


def _ast_to_sympy(node: ast.AST, symbols: dict):
    import sympy

    if isinstance(node, ast.Expression):
        return _ast_to_sympy(node.body, symbols)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed")
        return sympy.sympify(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sympy.pi
        if node.id == "E":
            return sympy.E
        if node.id == "I":
            return sympy.I
        if node.id in symbols:
            return symbols[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.BinOp):
        left = _ast_to_sympy(node.left, symbols)
        right = _ast_to_sympy(node.right, symbols)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise ValueError("disallowed operator")
    if isinstance(node, ast.UnaryOp):
        operand = _ast_to_sympy(node.operand, symbols)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError("disallowed unary operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("disallowed call")
        fname = node.func.id
        if fname not in _ALLOWED_FUNCS:
            raise ValueError(f"disallowed function: {fname}")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [_ast_to_sympy(a, symbols) for a in node.args]
        func = sympy.Abs if fname == "abs" else getattr(sympy, fname)
        return func(*args)
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def _format_number(value) -> str:
    import sympy

    if value in (sympy.zoo, sympy.oo, -sympy.oo) or (
        hasattr(value, "is_infinite") and value.is_infinite
    ):
        raise ZeroDivisionError("division by zero")
    if value.is_Integer:
        return str(int(value))
    if value.is_Rational:
        return f"{value.p}/{value.q}"
    val = sympy.N(value, 11)
    fval = float(val)
    if fval == int(fval):
        return str(int(fval))
    text = f"{fval:.10f}".rstrip("0").rstrip(".")
    return text


def _format_solution(sol) -> str:
    import sympy

    if sol.is_Integer:
        return str(int(sol))
    if sol.is_Rational:
        return f"{sol.p}/{sol.q}"
    decimal = sympy.N(sol, 12)
    return f"{sol} ≈ {decimal}"


def _compute_solve(inner: str) -> dict:
    import sympy

    if inner.count("=") != 1:
        return {
            "ok": False,
            "error": "solve(...) requires exactly one '=' sign, e.g. solve(2*x + 3 = 11)",
        }
    lhs_str, rhs_str = inner.split("=")
    x = sympy.Symbol("x")
    lhs_tree = ast.parse(lhs_str.strip(), mode="eval")
    rhs_tree = ast.parse(rhs_str.strip(), mode="eval")
    lhs = _ast_to_sympy(lhs_tree.body, {"x": x})
    rhs = _ast_to_sympy(rhs_tree.body, {"x": x})
    solutions = sympy.solve(sympy.Eq(lhs, rhs), x)
    if not solutions:
        return {"ok": False, "error": "no solution found"}
    parts = [f"x = {_format_solution(sol)}" for sol in solutions]
    return {"ok": True, "result": " or ".join(parts)}


def compute(expression: str) -> dict:
    try:
        stripped = expression.strip()
        match = _SOLVE_RE.match(stripped)
        if match:
            return _compute_solve(match.group(1))
        tree = ast.parse(expression, mode="eval")
        value = _ast_to_sympy(tree.body, {})
        return {"ok": True, "result": _format_number(value)}
    except ZeroDivisionError:
        return {"ok": False, "error": "division by zero"}
    except SyntaxError as exc:
        return {"ok": False, "error": f"syntax error: {exc}"}
    except (ValueError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        return {"ok": False, "error": str(exc)}
    except MemoryError:
        return {"ok": False, "error": "expression exceeded available memory"}


def main() -> None:
    import psutil

    self_process = psutil.Process()
    baseline_rss = self_process.memory_info().rss
    sys.stdout.write(json.dumps({"ready": True, "baseline_rss": baseline_rss}) + "\n")
    sys.stdout.flush()

    line = sys.stdin.readline()
    try:
        request = json.loads(line)
        expression = request["expression"]
    except (json.JSONDecodeError, KeyError, TypeError):
        sys.stdout.write(json.dumps({"ok": False, "error": "malformed request"}) + "\n")
        sys.stdout.flush()
        return

    result = compute(expression)
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

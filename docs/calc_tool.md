# calc tool

`tutor.tools.calc_tool.evaluate(expression, *, timeout_s=1.0,
memory_limit_mb=64) -> dict` evaluates a whitelisted arithmetic/algebra
expression in a fresh subprocess and returns
`{"ok": true, "result": "..."}` or `{"ok": false, "error": "..."}`. It
never raises for malformed, dangerous, slow, or oversized input, and the
parent process never calls `eval()`/`exec()` on model-supplied text.

## Whitelist

The child process (`python -m tutor.tools.calc_worker`) parses the
expression with `ast.parse(..., mode="eval")` and walks the tree with an
explicit whitelist (`tutor/tools/calc_worker.py::_ast_to_sympy`) before
ever touching sympy:

- Node types: `Expression`, numeric `Constant` (int/float only, no
  strings/bytes/bools), `Name` (only `pi`, `E`, `I`, and — inside
  `solve(...)` only — `x`), `BinOp` (`+ - * / ** %`), `UnaryOp` (`+x`,
  `-x`), and `Call`.
- Callable names: `sqrt`, `sin`, `cos`, `tan`, `log`, `exp`, `abs`,
  `factorial`, `Rational`. No keyword arguments are accepted.
- Everything else (attribute access, subscripting, comprehensions,
  lambdas, string/name literals other than the constants above, unknown
  calls, `import`, assignment, etc.) is rejected as `ok: false` before any
  sympy object is built.

One extra expression form is accepted: `solve(<linear/quadratic equation
in x>)`, e.g. `solve(2*x + 3 = 11)` -> `"x = 4"`, or
`solve(x**2 - 5*x + 6 = 0)` -> `"x = 2 or x = 3"`. Exactly one `=` sign is
required and the only free symbol allowed is `x`. When a root is not
already a plain integer or exact fraction, a decimal approximation is
appended: `solve(x**2 - 2 = 0)` -> `"x = sqrt(2) ≈ 1.41421356237"`.

## Limits

- **Length**: expressions over 200 characters are rejected without
  spawning a subprocess (spec §9.2).
- **Wall clock**: 1.0s by default (`timeout_s`). Enforced by the parent
  polling for the child's one line of JSON output; on timeout the child
  is killed (`Process.kill()`, portable) and `ok: false` is returned. The
  parent process is never blocked past the deadline and survives to
  evaluate the next expression.
- **Memory**: 64 MB by default (`memory_limit_mb`), enforced as **growth
  above the child's own post-import baseline RSS**, not as an absolute
  ceiling. Windows has no `resource.RLIMIT_AS`, so the parent instead
  polls the child's RSS via `psutil` every 20ms and kills it if RSS grows
  by more than the cap above the baseline the child reported right after
  `import sympy` (see the "ready" message in
  `tutor/tools/calc_worker.py`). An absolute-from-zero 64 MB ceiling would
  be unreliable across machines, since a freshly-imported sympy
  interpreter's own baseline RSS varies by machine, Python build, and
  installed sympy version. On Linux, `tutor.platform_.linux.
  limit_child_process` additionally sets a real `RLIMIT_AS` as a
  `preexec_fn` for defense in depth (best-effort; not exercised by this
  test suite, which runs on Windows).

## Measured on this development machine

(Windows 11, Python 3.12, sympy installed; single-sample, wall-clock
measurements, not a benchmark — see `docs/agent_loop.md` for how this fits
into a turn.)

- Child post-import baseline RSS (right after `import sympy`, before
  reading the expression): **~19.2 MB** (`19185664` bytes observed), well
  under the 64 MB growth cap.
- A typical call (`evaluate("4871*392")`, cold subprocess start each
  time): **~0.55s**, dominated by Python interpreter + sympy import
  startup rather than the arithmetic itself. This is why `timeout_s`
  defaults to 1.0s rather than something much smaller: the 1s budget is
  mostly startup headroom, not compute headroom.

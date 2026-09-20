"""Sandboxed arithmetic evaluation for the tutor's `calc` tool.

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §9.2 (calc
contract: `{ "ok": true, "result": "..." }` / `{ "ok": false, "error": "..." }`,
sympy whitelist, <=200 chars, subprocess with 1 s wall clock + 64 MB memory
cap) and docs/plan/offline_tutor_implementation_plan.md WP-C2 (Windows has
no `resource` module: subprocess timeout for wall clock + psutil polling
for memory; same interface so Linux can add RLIMIT_AS later).

Every call to :func:`evaluate` spawns a fresh child process
(``python -m tutor.tools.calc_worker``) that does the actual parsing and
evaluation; the parent process never calls ``eval()``/``exec()`` on the
model-supplied expression, and a hung or memory-hungry child cannot corrupt
or block the parent (see ``test_evaluate_runs_in_a_subprocess_not_the_
parent_interpreter`` and the "artificially slow" test in
tests/test_calc_tool.py).

Memory cap semantics
---------------------
The spec's 64 MB cap cannot be enforced as an absolute ceiling on the
child's total RSS: a freshly-started Python interpreter that has merely
``import sympy``'d sits at some baseline RSS that varies by machine and
can itself be a meaningful fraction of (or, on some machines, exceed) 64
MB, well before it has looked at the expression at all (see
docs/calc_tool.md for a measured baseline on this development machine).
Enforcing an absolute 64 MB from zero would therefore reject expressions
that never come close to using 64 MB of *working* memory.

Instead, the child reports its own post-import RSS as a "ready" message
before it reads the expression (see tutor/tools/calc_worker.py), and this
module enforces the 64 MB cap as **growth above that baseline**: if the
child's RSS grows by more than ``memory_limit_mb`` above its own
just-after-import baseline while evaluating, it is killed and treated as
an ``ok: false`` memory-cap error. Wall-clock (`timeout_s`, default 1.0s)
is enforced independently and unconditionally kills the child if exceeded.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time

import psutil

MAX_EXPRESSION_CHARS = 200
DEFAULT_TIMEOUT_S = 1.0
DEFAULT_MEMORY_LIMIT_MB = 64

_POLL_INTERVAL_S = 0.02
_STARTUP_TIMEOUT_S = 5.0


def _reader_thread(pipe, out_queue: queue.Queue) -> None:
    try:
        line = pipe.readline()
    except (OSError, ValueError):
        line = ""
    out_queue.put(line)


def _kill(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
        proc.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def evaluate(
    expression: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> dict:
    """Evaluate ``expression`` in a sandboxed subprocess.

    Returns ``{"ok": True, "result": "..."}`` on success or
    ``{"ok": False, "error": "..."}`` on any failure -- never raises for
    malformed, dangerous, slow, or oversized input.
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"ok": False, "error": "expression must be a non-empty string"}
    if len(expression) > MAX_EXPRESSION_CHARS:
        return {
            "ok": False,
            "error": f"expression exceeds the {MAX_EXPRESSION_CHARS} character limit",
        }

    deadline = time.monotonic() + timeout_s

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "tutor.tools.calc_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        return {"ok": False, "error": f"failed to start calc worker: {exc}"}

    try:
        ps_proc: psutil.Process | None = psutil.Process(proc.pid)
    except psutil.Error:
        ps_proc = None

    ready_queue: queue.Queue = queue.Queue()
    threading.Thread(
        target=_reader_thread, args=(proc.stdout, ready_queue), daemon=True
    ).start()

    startup_deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    try:
        ready_line = ready_queue.get(timeout=max(0.0, startup_deadline - time.monotonic()))
    except queue.Empty:
        _kill(proc)
        return {"ok": False, "error": "calc worker failed to start"}

    baseline_rss = 0
    try:
        ready = json.loads(ready_line) if ready_line else {}
        baseline_rss = int(ready.get("baseline_rss", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        _kill(proc)
        return {"ok": False, "error": "calc worker did not report readiness"}

    try:
        proc.stdin.write(json.dumps({"expression": expression}) + "\n")
        proc.stdin.flush()
    except (OSError, ValueError) as exc:
        _kill(proc)
        return {"ok": False, "error": f"failed to send expression to calc worker: {exc}"}

    result_queue: queue.Queue = queue.Queue()
    threading.Thread(
        target=_reader_thread, args=(proc.stdout, result_queue), daemon=True
    ).start()

    memory_limit_bytes = memory_limit_mb * 1024 * 1024
    result_line = None
    while time.monotonic() < deadline:
        try:
            result_line = result_queue.get(timeout=_POLL_INTERVAL_S)
            break
        except queue.Empty:
            pass
        if ps_proc is not None:
            try:
                rss = ps_proc.memory_info().rss
            except psutil.Error:
                continue
            if rss - baseline_rss > memory_limit_bytes:
                _kill(proc)
                return {
                    "ok": False,
                    "error": f"expression exceeded the {memory_limit_mb} MB memory cap",
                }

    if not result_line:
        _kill(proc)
        return {"ok": False, "error": f"expression exceeded the {timeout_s}s time limit"}

    _kill(proc)
    try:
        result = json.loads(result_line)
    except json.JSONDecodeError:
        return {"ok": False, "error": "calc worker returned malformed output"}

    if result.get("ok"):
        return {"ok": True, "result": str(result.get("result", ""))}
    return {"ok": False, "error": str(result.get("error") or "unknown calc error")}

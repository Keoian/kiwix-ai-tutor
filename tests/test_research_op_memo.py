"""RED tests for the per-request (op, args) -> result memo (Fix 2,
docs/retrieval_latency_profile.md "Ranked candidate fixes" #2).

``_call_worker`` and ``_call_worker_multi`` do not yet accept a ``memo``
dict, so a caller cannot avoid re-sending an identical (op, kwargs) pair to
the worker within one request. These tests exercise the helpers directly
(no real worker/archive needed) with a small counting fake standing in for
``ZimWorker``.
"""

from __future__ import annotations

from typing import Any

from tutor.retrieval.research import _call_worker, _call_worker_multi
from tutor.retrieval.zim.worker import WorkerResult


class _CountingWorker:
    """Fake worker: records every op it actually executes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[tuple[str, Any], ...]]] = []
        self.fail_terms: set[str] = set()

    def _run_one(self, op: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((op, tuple(sorted(kwargs.items()))))
        if op == "estimated_matches" and kwargs.get("term") in self.fail_terms:
            return {"status": "error", "value": None, "error": "boom"}
        return {"status": "ok", "value": f"{op}:{kwargs}", "error": None}

    def request(self, op: str, *, deadline_s: float, **kwargs: Any) -> WorkerResult:
        if op == "multi":
            results = [self._run_one(sub_op, sub_kwargs) for sub_op, sub_kwargs in kwargs["ops"]]
            status = "ok" if all(r["status"] == "ok" for r in results) else "partial"
            return WorkerResult(status=status, value=results, error=None, elapsed_s=0.0)
        result = self._run_one(op, kwargs)
        return WorkerResult(
            status=result["status"], value=result["value"], error=result["error"], elapsed_s=0.0
        )


def test_call_worker_memoizes_duplicate_single_op_within_one_memo() -> None:
    worker = _CountingWorker()
    memo: dict[Any, Any] = {}
    r1 = _call_worker(worker, "estimated_matches", deadline_s=1.0, term="helium", memo=memo)
    r2 = _call_worker(worker, "estimated_matches", deadline_s=1.0, term="helium", memo=memo)
    assert r1.status == "ok" and r2.status == "ok"
    assert r1.value == r2.value
    assert len(worker.calls) == 1, worker.calls


def test_call_worker_without_memo_hits_worker_every_time() -> None:
    worker = _CountingWorker()
    _call_worker(worker, "estimated_matches", deadline_s=1.0, term="helium")
    _call_worker(worker, "estimated_matches", deadline_s=1.0, term="helium")
    assert len(worker.calls) == 2


def test_call_worker_multi_dedupes_repeat_op_across_two_batches() -> None:
    worker = _CountingWorker()
    memo: dict[Any, Any] = {}
    ops = [("estimated_matches", {"term": "helium"}), ("estimated_matches", {"term": "celsius"})]
    res1 = _call_worker_multi(worker, ops, deadline_s=1.0, memo=memo)
    res2 = _call_worker_multi(worker, ops, deadline_s=1.0, memo=memo)
    assert res1.status == "ok" and res2.status == "ok"
    assert res1.value == res2.value
    # Only the first batch should have actually reached the worker.
    assert len(worker.calls) == 2, worker.calls


def test_call_worker_multi_dedupes_identical_op_within_one_batch() -> None:
    worker = _CountingWorker()
    memo: dict[Any, Any] = {}
    ops = [
        ("estimated_matches", {"term": "helium"}),
        ("search_titles", {"query": "x", "limit": 3}),
        ("estimated_matches", {"term": "helium"}),
    ]
    res = _call_worker_multi(worker, ops, deadline_s=1.0, memo=memo)
    assert res.status == "ok"
    assert res.value[0] == res.value[2]
    assert len(worker.calls) == 2, worker.calls  # helium fetched once, search_titles once


def test_call_worker_multi_does_not_memoize_errors() -> None:
    worker = _CountingWorker()
    worker.fail_terms.add("helium")
    memo: dict[Any, Any] = {}
    ops = [("estimated_matches", {"term": "helium"})]
    res1 = _call_worker_multi(worker, ops, deadline_s=1.0, memo=memo)
    assert res1.value[0]["status"] == "error"
    worker.fail_terms.discard("helium")
    res2 = _call_worker_multi(worker, ops, deadline_s=1.0, memo=memo)
    assert res2.value[0]["status"] == "ok"
    assert len(worker.calls) == 2, "the failed call must not be memoized, so it is retried"


def test_call_worker_multi_scopes_memo_by_worker_identity() -> None:
    worker_a = _CountingWorker()
    worker_b = _CountingWorker()
    memo: dict[Any, Any] = {}
    ops = [("estimated_matches", {"term": "helium"})]
    _call_worker_multi(worker_a, ops, deadline_s=1.0, memo=memo)
    _call_worker_multi(worker_b, ops, deadline_s=1.0, memo=memo)
    assert len(worker_a.calls) == 1
    assert len(worker_b.calls) == 1, (
        "a different worker must not be served from another worker's memo"
    )

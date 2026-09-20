"""RED tests for the ``multi`` batching op (docs/worker_batching_design.md).

``ZimWorker.multi()`` does not exist yet, and the worker child loop does not
handle the ``"multi"`` op yet, so every test here should fail (AttributeError
for the client helper tests, or a ``WorkerResult(status="error", ...)``
containing "unknown op: multi" for tests going through the raw ``request``
op) until the op is implemented. The research.py-level test should fail
because ``ResearchEngine`` still issues ~20 individual round-trips per
request instead of the batched ~4-6 target from the design note.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tutor.retrieval.registry import load_registry
from tutor.retrieval.research import ResearchEngine
from tutor.retrieval.snapshots import SnapshotStore
from tutor.retrieval.zim.worker import WorkerResult, ZimWorker

# Design note's batch length cap (docs/worker_batching_design.md).
_BATCH_CAP = 16


def _write_registry_toml(tmp_path: Path, fixture_zim: Path) -> Path:
    text = (
        "[[archive]]\n"
        'id = "tier1"\n'
        f'path = "{fixture_zim.as_posix()}"\n'
        "tier = 1\n"
        'kind = "encyclopedia"\n'
        "subjects = []\n"
        'storage = "ssd"\n'
    )
    toml_path = tmp_path / "registry.toml"
    toml_path.write_text(text, encoding="utf-8")
    return toml_path


def test_multi_returns_results_in_order(fixture_zim: Path) -> None:
    with ZimWorker(fixture_zim) as worker:
        result = worker.multi(
            [
                ("ping", {}),
                ("search_titles", {"query": "Pythagorean", "limit": 10}),
                ("estimated_matches", {"term": "Pythagoras"}),
            ],
            deadline_s=10.0,
        )
    assert isinstance(result, WorkerResult)
    assert result.status == "ok"
    assert len(result.value) == 3
    assert result.value[0]["status"] == "ok"
    assert result.value[1]["status"] == "ok"
    assert isinstance(result.value[1]["value"], list)
    assert result.value[2]["status"] == "ok"
    assert result.value[2]["value"] > 0


def test_multi_mixed_success_and_error_does_not_fail_whole_batch(fixture_zim: Path) -> None:
    with ZimWorker(fixture_zim) as worker:
        result = worker.multi(
            [
                ("ping", {}),
                ("fetch_entry", {"path": "does_not_exist_at_all"}),
                ("ping", {}),
            ],
            deadline_s=10.0,
        )
    assert result.status == "ok"
    assert result.value[0]["status"] == "ok"
    assert result.value[1]["status"] == "error"
    assert result.value[1]["error"]
    # A failing sub-op must not prevent the sub-op after it from running.
    assert result.value[2]["status"] == "ok"


def test_multi_empty_batch_returns_empty_list(fixture_zim: Path) -> None:
    with ZimWorker(fixture_zim) as worker:
        result = worker.multi([], deadline_s=10.0)
    assert result.status == "ok"
    assert result.value == []


def test_multi_deadline_expiry_mid_batch_returns_partial_with_completed_results(
    fixture_zim: Path,
) -> None:
    with ZimWorker(fixture_zim) as worker:
        result = worker.multi(
            [
                ("ping", {}),
                ("sleep", {"seconds": 5}),
                ("ping", {}),
            ],
            deadline_s=1.0,
        )
    # The first sub-op completes; the deadline is exhausted by the sleep
    # (which itself either overruns and kills the child -- covered by the
    # existing outer poll timeout -- or, once the child-side budget check
    # lands, the third sub-op is skipped rather than run past budget).
    assert result.status in ("partial", "timeout")
    if result.status == "partial":
        assert result.value[0]["status"] == "ok"
        assert result.value[-1]["status"] == "error"
        assert "deadline" in result.value[-1]["error"]


def test_multi_batch_size_cap_rejected(fixture_zim: Path) -> None:
    with ZimWorker(fixture_zim) as worker:
        with pytest.raises(ValueError):
            worker.multi([("ping", {})] * (_BATCH_CAP + 1), deadline_s=10.0)


def test_client_helper_multi_is_a_thin_wrapper_over_request(fixture_zim: Path) -> None:
    # The client helper must exist as a first-class method, not something
    # callers reimplement by hand-packing ("multi", {"ops": ...}) kwargs.
    with ZimWorker(fixture_zim) as worker:
        assert hasattr(worker, "multi")
        via_helper = worker.multi([("ping", {})], deadline_s=10.0)
        via_raw = worker.request("multi", deadline_s=10.0, ops=[("ping", {})])
    assert via_helper.status == via_raw.status == "ok"
    assert via_helper.value == via_raw.value


# ---------------------------------------------------------------------------
# research.py level: round-trip count for a standard request
# ---------------------------------------------------------------------------


class _CountingWorker:
    """Wraps a real worker/fake to count ``request()`` calls (round-trips)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.call_count = 0

    def request(self, op: str, *, deadline_s: float, **kwargs: Any) -> WorkerResult:
        self.call_count += 1
        return self._inner.request(op, deadline_s=deadline_s, **kwargs)

    def interrupt(self) -> None:
        interrupt = getattr(self._inner, "interrupt", None)
        if callable(interrupt):
            interrupt()

    def close(self) -> None:
        self._inner.close()


# Design note target (docs/worker_batching_design.md): ~21 round-trips today
# collapse to 4-6 once the fixed-shape batches (initial fulltext+titles,
# entity estimated_matches, entity title+snippet, fetch_entry) are each one
# `multi` call. Generous upper bound so the test is about batching, not
# about pinning the exact conditional-fallback call count.
_TARGET_MAX_ROUND_TRIPS = 8


def test_research_request_round_trips_drop_to_target(tmp_path, fixture_zim: Path) -> None:
    registry_toml = _write_registry_toml(tmp_path, fixture_zim)
    registry = load_registry(registry_toml)
    snapshot_store = SnapshotStore(tmp_path / "snapshots.sqlite3")
    counters: list[_CountingWorker] = []

    def _worker_factory(path: Path) -> _CountingWorker:
        counting = _CountingWorker(ZimWorker(path))
        counters.append(counting)
        return counting

    engine = ResearchEngine(
        registry,
        snapshot_store=snapshot_store,
        cache_dir=tmp_path / "cache",
        worker_factory=_worker_factory,
    )
    try:
        engine.research("What is the Pythagorean theorem?")
    finally:
        snapshot_store.close()

    total_round_trips = sum(c.call_count for c in counters)
    assert total_round_trips <= _TARGET_MAX_ROUND_TRIPS, (
        f"expected batching to bring round-trips to <= {_TARGET_MAX_ROUND_TRIPS}, "
        f"got {total_round_trips}"
    )

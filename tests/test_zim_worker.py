"""RED tests for ``tutor.retrieval.zim.worker`` (WP-B3).

A libzim child-process worker, spawned with multiprocessing's "spawn"
context (never "fork"), that owns operations which could wedge or block
(Xapian search, entry reads) so the app can enforce soft/hard deadlines
per docs/plan/offline_tutor_spec_v0.3.md §7.4 and hard-kill a stuck child.
"""

from __future__ import annotations

from pathlib import Path

import psutil
import pytest

from tutor.retrieval.zim.search import search_fulltext, search_titles
from tutor.retrieval.zim.worker import WorkerClosed, WorkerResult, ZimWorker


def _own_children(pid: int | None = None) -> list[psutil.Process]:
    return psutil.Process().children(recursive=True)


def test_worker_module_uses_spawn_context_never_fork() -> None:
    import tutor.retrieval.zim.worker as worker_module

    source = Path(worker_module.__file__).read_text(encoding="utf-8")
    assert 'get_context("spawn")' in source
    assert "fork" not in source


def test_worker_ping_returns_ok(fixture_zim: Path) -> None:
    with ZimWorker(fixture_zim) as worker:
        result = worker.request("ping", deadline_s=10.0)
        assert isinstance(result, WorkerResult)
        assert result.status == "ok"


def test_worker_search_fulltext_matches_inprocess_search(fixture_zim: Path) -> None:
    from libzim.reader import Archive

    archive = Archive(str(fixture_zim))
    expected = search_fulltext(archive, "Pythagorean theorem", limit=20)

    with ZimWorker(fixture_zim) as worker:
        result = worker.request(
            "search_fulltext", deadline_s=10.0, query="Pythagorean theorem", limit=20
        )
    assert result.status == "ok"
    assert result.value == expected


def test_worker_search_fulltext_snippet_top_n_reaches_child(fixture_zim: Path) -> None:
    """Baseline v9 candidate B: the ``snippet_top_n`` kwarg must reach the
    child's ``search_fulltext`` call, not just live in the parent."""
    from libzim.reader import Archive

    archive = Archive(str(fixture_zim))
    expected = search_fulltext(archive, "the", limit=10, snippet_top_n=2)

    with ZimWorker(fixture_zim) as worker:
        result = worker.request(
            "search_fulltext", deadline_s=10.0, query="the", limit=10, snippet_top_n=2
        )
    assert result.status == "ok"
    assert result.value == expected
    non_empty = [h for h in result.value if h.snippet]
    assert len(non_empty) <= 2


def test_worker_search_titles_matches_inprocess_search(fixture_zim: Path) -> None:
    from libzim.reader import Archive

    archive = Archive(str(fixture_zim))
    expected = search_titles(archive, "Pythagorean", limit=10)

    with ZimWorker(fixture_zim) as worker:
        result = worker.request(
            "search_titles", deadline_s=10.0, query="Pythagorean", limit=10
        )
    assert result.status == "ok"
    assert result.value == expected


def test_worker_estimated_matches_matches_inprocess_search(fixture_zim: Path) -> None:
    from libzim.reader import Archive

    from tutor.retrieval.zim.search import estimated_matches

    archive = Archive(str(fixture_zim))
    expected = estimated_matches(archive, "Pythagoras")

    with ZimWorker(fixture_zim) as worker:
        result = worker.request("estimated_matches", deadline_s=10.0, term="Pythagoras")
    assert result.status == "ok"
    assert result.value == expected
    assert expected > 0


def test_worker_fetch_entry_returns_plain_data(fixture_zim: Path) -> None:
    with ZimWorker(fixture_zim) as worker:
        result = worker.request("fetch_entry", deadline_s=10.0, path="pythagorean_theorem")
    assert result.status == "ok"
    assert result.value.path == "pythagorean_theorem"
    assert "Pythagorean" in result.value.title


def test_worker_result_fields(fixture_zim: Path) -> None:
    with ZimWorker(fixture_zim) as worker:
        result = worker.request("ping", deadline_s=10.0)
    assert result.status in {"ok", "partial", "timeout", "error"}
    assert result.error is None
    assert isinstance(result.elapsed_s, float)
    assert result.elapsed_s >= 0.0


def test_worker_deadline_exceeded_returns_timeout_and_kills_child(fixture_zim: Path) -> None:
    with ZimWorker(fixture_zim) as worker:
        worker.request("ping", deadline_s=10.0)
        old_pid = worker.pid
        assert old_pid is not None

        result = worker.request("sleep", deadline_s=0.5, seconds=5)
        assert result.status == "timeout"

        # The old child must no longer be alive.
        assert not psutil.pid_exists(old_pid) or not _pid_is_worker_child(old_pid)

        # A fresh request succeeds and lazily spawns a new child.
        follow_up = worker.request("ping", deadline_s=10.0)
        assert follow_up.status == "ok"
        new_pid = worker.pid
        assert new_pid is not None
        assert new_pid != old_pid

        # Exactly one live worker child of this process.
        my_children = psutil.Process().children(recursive=True)
        alive_pids = {p.pid for p in my_children if p.is_running()}
        assert old_pid not in alive_pids
        assert new_pid in alive_pids


def _pid_is_worker_child(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_worker_crash_returns_error_and_restarts(fixture_zim: Path) -> None:
    with ZimWorker(fixture_zim) as worker:
        worker.request("ping", deadline_s=10.0)
        result = worker.request("crash", deadline_s=10.0)
        assert result.status == "error"

        follow_up = worker.request("ping", deadline_s=10.0)
        assert follow_up.status == "ok"


def test_worker_close_leaves_no_live_child(fixture_zim: Path) -> None:
    worker = ZimWorker(fixture_zim)
    worker.start()
    worker.request("ping", deadline_s=10.0)
    pid = worker.pid
    worker.close()

    assert pid is not None
    assert not _pid_is_worker_child(pid)


def test_worker_close_is_idempotent(fixture_zim: Path) -> None:
    worker = ZimWorker(fixture_zim)
    worker.start()
    worker.close()
    worker.close()  # must not raise


def test_worker_request_after_close_raises_worker_closed(fixture_zim: Path) -> None:
    worker = ZimWorker(fixture_zim)
    worker.start()
    worker.close()
    with pytest.raises(WorkerClosed):
        worker.request("ping", deadline_s=10.0)


def test_worker_invalid_archive_returns_error_and_does_not_hang(not_a_zim: Path) -> None:
    worker = ZimWorker(not_a_zim)
    try:
        result = worker.request("ping", deadline_s=10.0)
        assert result.status == "error"
        assert result.error
    finally:
        worker.close()

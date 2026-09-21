"""RED tests for tutor.retrieval.snapshots (WP-B5, citation snapshot store).

Module does not exist yet; every test here should fail with
ModuleNotFoundError until implemented. See docs/plan/
offline_tutor_implementation_plan.md WP-B5 ("Snapshot store. Acceptance:
exact highlight after disposable caches are cleared") and
offline_tutor_spec_v0.3.md line ~294 ("citation snapshots stored separately
from disposable caches") and line ~298 ("source viewer highlights the exact
sentence(s) ... using the passage's stored character offsets").
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path

from tutor.retrieval.snapshots import SnapshotStore, locate_in_text

FINGERPRINT = "abc123ff" * 8


@dataclasses.dataclass(frozen=True)
class _FakePassage:
    """Minimal stand-in for tutor.retrieval.hybrid.passages.Passage.

    Only the attributes SnapshotStore.put is documented to read are set,
    so this test file has no dependency on the passages module (owned by
    a concurrent worker) or its exact field set.
    """

    passage_id: str
    archive_id: str
    path: str
    title: str
    heading_path: tuple
    start: int
    end: int
    text: str


def _make_passage(passage_id="0" * 32, text="Erdős worked here. \U0001F600") -> _FakePassage:
    return _FakePassage(
        passage_id=passage_id,
        archive_id="fixture_en_school",
        path="erdos_number",
        title="Erdős number",
        heading_path=("Definition",),
        start=10,
        end=10 + len(text),
        text=text,
    )


def test_snapshot_store_uses_wal_mode(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    with SnapshotStore(db_path) as store:
        mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


def test_falls_back_to_delete_journal_when_wal_pragma_fails(
    tmp_path: Path, monkeypatch
):
    """WAL needs a memory-mapped -shm file; on some ephemeral/CI filesystems
    that mmap raises ``sqlite3.OperationalError: disk I/O error`` even though
    ordinary reads/writes work fine (observed on ubuntu-latest GitHub
    Actions runners). The store must not propagate that failure -- it should
    fall back to the classic rollback journal and still be usable.
    """
    import sqlite3

    class _FlakyConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.strip().upper() == "PRAGMA JOURNAL_MODE=WAL":
                raise sqlite3.OperationalError("disk I/O error")
            return super().execute(sql, *args, **kwargs)

    real_connect = sqlite3.connect

    def _connect(*args, **kwargs):
        kwargs["factory"] = _FlakyConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _connect)

    db_path = tmp_path / "snapshots.sqlite3"
    with SnapshotStore(db_path) as store:
        mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "delete"
        passage = _FakePassage(
            passage_id="p1",
            archive_id="a1",
            path="A/x",
            title="T",
            heading_path=("H",),
            start=0,
            end=3,
            text="abc",
        )
        store.put(passage, fingerprint_digest=FINGERPRINT)
        assert store.get("p1").text == "abc"


def test_retries_through_a_transient_disk_io_error(tmp_path: Path, monkeypatch):
    """A single transient "disk I/O error" (observed on ubuntu-latest CI,
    consistent with a runner disk burst-credit throttle rather than a real
    fault) must not fail store construction -- it should retry and succeed.
    """
    import sqlite3

    calls = {"n": 0}
    real_execute = sqlite3.Connection.execute

    class _FlakyOnceConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.strip().upper().startswith("CREATE TABLE"):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise sqlite3.OperationalError("disk I/O error")
            return real_execute(self, sql, *args, **kwargs)

    real_connect = sqlite3.connect

    def _connect(*args, **kwargs):
        kwargs["factory"] = _FlakyOnceConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _connect)
    monkeypatch.setattr("tutor.retrieval._sqlite_retry.time.sleep", lambda _seconds: None)

    db_path = tmp_path / "snapshots.sqlite3"
    with SnapshotStore(db_path) as store:
        assert calls["n"] == 2
        passage = _FakePassage(
            passage_id="p1",
            archive_id="a1",
            path="A/x",
            title="T",
            heading_path=("H",),
            start=0,
            end=3,
            text="abc",
        )
        store.put(passage, fingerprint_digest=FINGERPRINT)
        assert store.get("p1").text == "abc"


def test_put_then_get_round_trips(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    passage = _make_passage()
    with SnapshotStore(db_path) as store:
        store.put(passage, fingerprint_digest=FINGERPRINT)
        snap = store.get(passage.passage_id)
    assert snap is not None
    assert snap.passage_id == passage.passage_id
    assert snap.archive_id == passage.archive_id
    assert snap.path == passage.path
    assert snap.title == passage.title
    assert snap.heading_path == passage.heading_path
    assert snap.start == passage.start
    assert snap.end == passage.end
    assert snap.text == passage.text
    assert snap.fingerprint_digest == FINGERPRINT
    assert snap.created_at is not None


def test_get_missing_returns_none(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    with SnapshotStore(db_path) as store:
        assert store.get("f" * 32) is None


def test_put_is_idempotent(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    passage = _make_passage()
    with SnapshotStore(db_path) as store:
        store.put(passage, fingerprint_digest=FINGERPRINT)
        store.put(passage, fingerprint_digest=FINGERPRINT)
        # No error, and exactly one row's worth of state retrievable.
        snap = store.get(passage.passage_id)
    assert snap is not None
    assert snap.text == passage.text


def test_exact_highlight_survives_close_and_reopen(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    passage = _make_passage(text="The exact cited sentence, unicode: éèê.")

    store = SnapshotStore(db_path)
    store.put(passage, fingerprint_digest=FINGERPRINT)
    store.close()

    # Simulate disposable caches cleared: nothing else touches db_path.
    reopened = SnapshotStore(db_path)
    snap = reopened.get(passage.passage_id)
    reopened.close()

    assert snap is not None
    assert snap.text == passage.text
    assert (snap.start, snap.end) == (passage.start, passage.end)


def test_unicode_text_round_trips_exactly(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    text = "combining é, emoji \U0001F600, math a²+b²=c², Erdős"
    passage = _make_passage(passage_id="1" * 32, text=text)
    with SnapshotStore(db_path) as store:
        store.put(passage, fingerprint_digest=FINGERPRINT)
        snap = store.get(passage.passage_id)
    assert snap is not None
    assert snap.text == text


def test_usable_from_a_second_thread(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    passage = _make_passage(passage_id="2" * 32)
    errors: list[Exception] = []

    with SnapshotStore(db_path) as store:
        def writer():
            try:
                store.put(passage, fingerprint_digest=FINGERPRINT)
            except Exception as exc:  # pragma: no cover - surfaced via errors
                errors.append(exc)

        t = threading.Thread(target=writer)
        t.start()
        t.join(timeout=10)
        assert not errors

        result_holder = {}

        def reader():
            try:
                result_holder["snap"] = store.get(passage.passage_id)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        t2 = threading.Thread(target=reader)
        t2.start()
        t2.join(timeout=10)
        assert not errors

    assert result_holder["snap"] is not None
    assert result_holder["snap"].text == passage.text


# ---------------------------------------------------------------------------
# locate_in_text
# ---------------------------------------------------------------------------


def test_locate_in_text_returns_stored_span_when_unchanged(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    passage = _make_passage(text="stable text here")
    with SnapshotStore(db_path) as store:
        store.put(passage, fingerprint_digest=FINGERPRINT)
        snap = store.get(passage.passage_id)

    current_text = "prefix " * 0 + ("x" * snap.start) + snap.text + "suffix text"
    result = locate_in_text(snap, current_text)
    assert result == (snap.start, snap.end)
    assert current_text[result[0]:result[1]] == snap.text


def test_locate_in_text_finds_text_elsewhere_when_span_shifted(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    passage = _make_passage(text="a distinctive phrase to find")
    with SnapshotStore(db_path) as store:
        store.put(passage, fingerprint_digest=FINGERPRINT)
        snap = store.get(passage.passage_id)

    shifted_text = "some new prefix content that pushes things " + snap.text + " and a suffix"
    result = locate_in_text(snap, shifted_text)
    assert result is not None
    start, end = result
    assert shifted_text[start:end] == snap.text


def test_locate_in_text_returns_none_when_absent(tmp_path: Path):
    db_path = tmp_path / "snapshots.sqlite3"
    passage = _make_passage(text="text that will vanish")
    with SnapshotStore(db_path) as store:
        store.put(passage, fingerprint_digest=FINGERPRINT)
        snap = store.get(passage.passage_id)

    result = locate_in_text(snap, "totally unrelated content, nothing matches here")
    assert result is None

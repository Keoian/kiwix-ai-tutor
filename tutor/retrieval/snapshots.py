"""Citation snapshot store (WP-B5).

Ours: no donor code. A passage's exact cited text and offsets are stored
here, separately from any disposable cache, so a citation can still be
highlighted exactly after caches are cleared or the underlying archive is
re-indexed. See docs/bundle_and_passages.md for the schema and
``locate_in_text``'s fallback behaviour.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tutor.retrieval._sqlite_retry import execute_with_retry as _execute_with_retry

_HEADING_PATH_SEP = "\x1f"


@dataclass(frozen=True)
class Snapshot:
    """A stored citation snapshot, as returned by :meth:`SnapshotStore.get`."""

    passage_id: str
    archive_id: str
    path: str
    title: str
    heading_path: tuple[str, ...]
    start: int
    end: int
    text: str
    fingerprint_digest: str
    created_at: str


class SnapshotStore:
    """SQLite-backed store of citation snapshots.

    Opens the database in WAL mode with ``check_same_thread=False`` plus an
    internal lock, so a single store instance is safe to share across
    threads (one connection, serialized access) -- the shape the tests in
    ``tests/test_snapshots.py`` exercise directly via ``store.connection``.
    """

    def __init__(self, db_path: Path) -> None:
        self._lock = threading.Lock()
        self.connection = sqlite3.connect(str(Path(db_path)), check_same_thread=False)
        with self._lock:
            # WAL needs a memory-mapped -shm file; on some ephemeral/CI
            # filesystems (e.g. certain tmpfs configurations) that mmap can
            # fail with "disk I/O error" even though plain reads/writes are
            # fine. Fall back to the classic rollback journal there rather
            # than let every subsequent write on this connection fail.
            try:
                _execute_with_retry(self.connection, "PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                _execute_with_retry(self.connection, "PRAGMA journal_mode=DELETE")
            _execute_with_retry(
                self.connection,
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    passage_id TEXT PRIMARY KEY,
                    archive_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    start INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    fingerprint_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            self.connection.commit()

    def put(self, passage: Any, *, fingerprint_digest: str) -> None:
        """Insert or overwrite the snapshot for ``passage.passage_id``.

        ``passage`` needs only the attributes read below (``passage_id``,
        ``archive_id``, ``path``, ``title``, ``heading_path``, ``start``,
        ``end``, ``text``) -- this module has no import-time dependency on
        ``tutor.retrieval.hybrid.passages.Passage``.
        """
        heading_path_str = _HEADING_PATH_SEP.join(passage.heading_path)
        created_at = datetime.now(UTC).isoformat()
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO snapshots
                    (passage_id, archive_id, path, title, heading_path,
                     start, end_offset, text, fingerprint_digest, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(passage_id) DO UPDATE SET
                    archive_id = excluded.archive_id,
                    path = excluded.path,
                    title = excluded.title,
                    heading_path = excluded.heading_path,
                    start = excluded.start,
                    end_offset = excluded.end_offset,
                    text = excluded.text,
                    fingerprint_digest = excluded.fingerprint_digest
                """,
                (
                    passage.passage_id,
                    passage.archive_id,
                    passage.path,
                    passage.title,
                    heading_path_str,
                    passage.start,
                    passage.end,
                    passage.text,
                    fingerprint_digest,
                    created_at,
                ),
            )
            self.connection.commit()

    def get(self, passage_id: str) -> Snapshot | None:
        with self._lock:
            row = self.connection.execute(
                """
                SELECT passage_id, archive_id, path, title, heading_path,
                       start, end_offset, text, fingerprint_digest, created_at
                FROM snapshots WHERE passage_id = ?
                """,
                (passage_id,),
            ).fetchone()
        if row is None:
            return None
        heading_path = tuple(row[4].split(_HEADING_PATH_SEP)) if row[4] else ()
        return Snapshot(
            passage_id=row[0],
            archive_id=row[1],
            path=row[2],
            title=row[3],
            heading_path=heading_path,
            start=row[5],
            end=row[6],
            text=row[7],
            fingerprint_digest=row[8],
            created_at=row[9],
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SnapshotStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def locate_in_text(snapshot: Snapshot, current_text: str) -> tuple[int, int] | None:
    """Find ``snapshot.text`` in ``current_text``, preferring the stored span.

    Returns ``(start, end)`` such that ``current_text[start:end] ==
    snapshot.text``: the stored ``(start, end)`` when it still matches
    (the common case -- nothing about the archive changed), otherwise the
    first occurrence found by a plain substring search (the archive was
    re-rendered and the citation shifted but the sentence survives), or
    ``None`` when the text is simply gone.
    """
    if current_text[snapshot.start : snapshot.end] == snapshot.text:
        return (snapshot.start, snapshot.end)
    index = current_text.find(snapshot.text)
    if index == -1:
        return None
    return (index, index + len(snapshot.text))

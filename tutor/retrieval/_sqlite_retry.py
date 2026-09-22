"""Shared retry helper for SQLite store initialization.

Observed on ubuntu-latest GitHub Actions runners: sqlite3 can raise
``sqlite3.OperationalError: disk I/O error`` from an ordinary
``CREATE TABLE`` / ``PRAGMA`` on a brand-new, empty database file, with no
reproducible code-level cause -- consistent with the runner's ephemeral
disk hitting a transient IOPS/burst-credit throttle under the heavy small
random I/O hundreds of short-lived per-test SQLite files produce, rather
than anything wrong with the statement itself. A bounded, exponentially
backed-off retry clears it without masking a genuinely persistent failure
(which keeps raising once the retries are exhausted).

Used by ``tutor.retrieval.snapshots.SnapshotStore``,
``tutor.app.profiles.ProfileStore``, and ``tutor.app.lesson_state.LessonStore``
so all three SQLite-backed stores get the same protection.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_TRANSIENT_IO_RETRIES = 9
_TRANSIENT_IO_BASE_DELAY_S = 0.2
_TRANSIENT_IO_MAX_DELAY_S = 5.0

# Busy handler wait before a locked-DB access raises SQLITE_BUSY (ms).
_BUSY_TIMEOUT_MS = 5000


def execute_with_retry(connection: sqlite3.Connection, sql: str) -> None:
    """Run ``connection.execute(sql)``, retrying on a transient disk I/O
    error with exponential backoff (capped), up to ``_TRANSIENT_IO_RETRIES``
    attempts total. Any other error, or the last attempt's error, propagates.
    """
    delay = _TRANSIENT_IO_BASE_DELAY_S
    for attempt in range(_TRANSIENT_IO_RETRIES):
        try:
            connection.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if "disk i/o error" not in str(exc).lower() or attempt == _TRANSIENT_IO_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _TRANSIENT_IO_MAX_DELAY_S)


def connect_store(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection for one of the tutor's small backing stores
    (profiles, lesson_state, snapshots) with I/O-pressure-reducing PRAGMAs
    applied consistently, then return it ready for the caller's own
    ``CREATE TABLE`` / schema setup.

    ``synchronous=NORMAL`` is the important one: the default ``FULL`` fsyncs on
    every commit, and that per-commit fsync -- multiplied by the hundreds of
    short-lived per-test databases the suite creates -- is the small-random-I/O
    that the ubuntu-latest runner's ephemeral disk throttles into
    ``SQLITE_IOERR`` (see module docstring). With WAL journalling, ``NORMAL``
    is crash-safe against application crashes and only risks losing the last
    transactions on an OS/power loss -- acceptable for these recreatable
    cache/session stores, and a real win on the delivery laptop's slow disk.
    ``busy_timeout`` makes a briefly-locked DB wait rather than raise
    ``SQLITE_BUSY``. The WAL PRAGMA still falls back to DELETE on a filesystem
    that cannot support it (the pre-existing behaviour, now centralised here).
    """
    connection = sqlite3.connect(str(Path(db_path)), check_same_thread=False)
    execute_with_retry(connection, f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    try:
        execute_with_retry(connection, "PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        execute_with_retry(connection, "PRAGMA journal_mode=DELETE")
    execute_with_retry(connection, "PRAGMA synchronous=NORMAL")
    return connection

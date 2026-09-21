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

_TRANSIENT_IO_RETRIES = 6
_TRANSIENT_IO_BASE_DELAY_S = 0.2
_TRANSIENT_IO_MAX_DELAY_S = 5.0


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

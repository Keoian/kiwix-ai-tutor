"""The three SQLite-backed stores (profiles, lesson_state, snapshots) all
open their connections through one shared helper so they get the same
I/O-pressure-reducing PRAGMAs. This closes the ubuntu-latest CI flake where
``sqlite3.OperationalError: disk I/O error`` outlasts the init retry backoff
(``tutor.retrieval._sqlite_retry``): the default ``synchronous=FULL`` fsyncs
on every commit, and that per-commit fsync under hundreds of short-lived
per-test DB files is exactly the small-random-I/O the runner's ephemeral
disk throttles. ``synchronous=NORMAL`` drops fsyncs to WAL-checkpoint only;
``busy_timeout`` keeps a briefly-locked DB from raising instead of waiting.
It also helps the real product on the delivery laptop's slow disk.
"""

from __future__ import annotations

from pathlib import Path

from tutor.retrieval._sqlite_retry import connect_store


def _pragma(conn, name: str):
    return conn.execute(f"PRAGMA {name}").fetchone()[0]


def test_connect_store_sets_reduced_io_pragmas(tmp_path: Path) -> None:
    conn = connect_store(tmp_path / "store.db")
    try:
        # synchronous=NORMAL is the integer 1 (0=OFF, 1=NORMAL, 2=FULL).
        assert _pragma(conn, "synchronous") == 1
        # busy_timeout is echoed back in milliseconds.
        assert _pragma(conn, "busy_timeout") == 5000
        # WAL where the filesystem supports it (tmp_path is ordinary ext4/tmpfs).
        assert _pragma(conn, "journal_mode").lower() == "wal"
    finally:
        conn.close()


def test_connect_store_returns_usable_connection(tmp_path: Path) -> None:
    conn = connect_store(tmp_path / "store.db")
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t (v) VALUES (?)", ("x",))
        conn.commit()
        assert conn.execute("SELECT v FROM t").fetchone()[0] == "x"
    finally:
        conn.close()

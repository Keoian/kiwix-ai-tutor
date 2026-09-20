"""Out-of-process ZIM worker enforcing hard deadlines on libzim calls.

The child process is started via multiprocessing's ``get_context("spawn")``
(never the copy-on-write child-process mode some platforms default to) so
behavior is identical on Windows and Linux. The child target is a plain
module-level function that opens the archive once and then serves requests
from a pipe in a loop. The parent enforces the deadline itself with
``Connection.poll(timeout)``: on a timeout it hard-kills the child and joins
it, marks the worker dead, and lazily starts a fresh child on the next
request. Only plain, picklable data ever crosses the pipe — libzim objects
(``Archive``, ``Entry``, ``Item``) never leave the child.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from tutor.retrieval.zim.search import fetch_entry, search_fulltext, search_titles

_CONTEXT = mp.get_context("spawn")

# Ops handled directly by the child loop without touching libzim, used by
# tests to exercise the deadline/crash/restart machinery in isolation.
_TEST_ONLY_OPS = {"sleep", "crash"}


class WorkerClosed(Exception):
    """Raised by :meth:`ZimWorker.request` after :meth:`ZimWorker.close`."""


@dataclass(frozen=True)
class WorkerResult:
    """Outcome of one :meth:`ZimWorker.request` call."""

    status: str  # "ok" | "partial" | "timeout" | "error"
    value: Any
    error: str | None
    elapsed_s: float


def _child_main(archive_path_str: str, conn: Connection) -> None:
    """Child process entry point: open the archive once, then serve requests.

    Never raises out of the process on a bad request or a missing/invalid
    archive; each failure is reported back as an ``("error", ...)`` reply so
    the parent never has to guess why a request failed. The ``"crash"``
    test-only op is the sole intentional exception: it ends the process
    itself so the parent observes a broken pipe.
    """
    archive: Any = None
    archive_error: str | None = None
    try:
        from libzim.reader import Archive

        archive = Archive(archive_path_str)
    except Exception as exc:  # noqa: BLE001 - reported to parent, not raised
        archive_error = str(exc)

    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        op, kwargs = message

        if op == "crash":
            # Test-only op: simulate a wedged/misbehaving child by ending
            # the process without a reply, so the parent sees a closed pipe.
            raise SystemExit(1)

        if op == "sleep":
            # Test-only op: hold the child busy past the parent's deadline.
            time.sleep(float(kwargs.get("seconds", 0.0)))
            conn.send(("ok", None, None))
            continue

        if archive is None:
            conn.send(("error", None, archive_error or "archive not available"))
            continue

        if op == "ping":
            conn.send(("ok", None, None))
            continue

        try:
            if op == "search_fulltext":
                value = search_fulltext(
                    archive, kwargs["query"], limit=kwargs.get("limit", 20)
                )
            elif op == "search_titles":
                value = search_titles(
                    archive, kwargs["query"], limit=kwargs.get("limit", 10)
                )
            elif op == "fetch_entry":
                value = fetch_entry(archive, kwargs["path"])
            else:
                conn.send(("error", None, f"unknown op: {op}"))
                continue
        except Exception as exc:  # noqa: BLE001 - reported to parent, not raised
            conn.send(("error", None, str(exc)))
            continue
        conn.send(("ok", value, None))


class ZimWorker:
    """Manages a single lazily-(re)started child process serving ``archive_path``."""

    def __init__(self, archive_path: Path) -> None:
        self._archive_path = Path(archive_path)
        self._process: Any = None
        self._conn: Connection | None = None
        self._closed = False

    def __enter__(self) -> "ZimWorker":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def pid(self) -> int | None:
        if self._process is not None and self._process.is_alive():
            return self._process.pid
        return None

    def start(self) -> None:
        if self._closed:
            raise WorkerClosed("Cannot start a closed ZimWorker.")
        if self._process is not None and self._process.is_alive():
            return
        parent_conn, child_conn = _CONTEXT.Pipe(duplex=True)
        process = _CONTEXT.Process(
            target=_child_main,
            args=(str(self._archive_path), child_conn),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process = process
        self._conn = parent_conn

    def _terminate_child(self) -> None:
        if self._process is not None:
            try:
                if self._process.is_alive():
                    self._process.kill()
                self._process.join(timeout=5.0)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self._process = None
        self._conn = None

    def request(self, op: str, *, deadline_s: float, **kwargs: Any) -> WorkerResult:
        if self._closed:
            raise WorkerClosed("Cannot request on a closed ZimWorker.")
        self.start()
        assert self._conn is not None
        started = time.monotonic()
        try:
            self._conn.send((op, kwargs))
        except (OSError, EOFError, BrokenPipeError) as exc:
            self._terminate_child()
            return WorkerResult(
                status="error", value=None, error=str(exc), elapsed_s=time.monotonic() - started
            )

        ready = self._conn.poll(deadline_s)
        if not ready:
            self._terminate_child()
            return WorkerResult(
                status="timeout",
                value=None,
                error=f"deadline of {deadline_s}s exceeded for op={op!r}",
                elapsed_s=time.monotonic() - started,
            )

        try:
            status, value, error = self._conn.recv()
        except (EOFError, OSError, ConnectionResetError) as exc:
            self._terminate_child()
            return WorkerResult(
                status="error", value=None, error=str(exc), elapsed_s=time.monotonic() - started
            )

        return WorkerResult(
            status=status, value=value, error=error, elapsed_s=time.monotonic() - started
        )

    def close(self) -> None:
        if self._closed:
            return
        self._terminate_child()
        self._closed = True

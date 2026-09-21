"""Session-scoped ZIM fixtures shared by WP-B1/WP-B2 tests.

The fixtures are built once per test session with ``libzim.writer.Creator``
(see ``tests/zim_fixtures.py``) so tests exercise real libzim binaries.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import zim_fixtures

_fd_growth: list[tuple[str, int]] = []


def _linux_fd_count() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def _linux_child_count() -> int:
    try:
        out = subprocess.run(
            ["ps", "--ppid", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        lines = [ln for ln in out.splitlines() if ln.strip()]
        return max(0, len(lines) - 1)
    except Exception:
        return -1


@pytest.fixture(autouse=True)
def _fd_growth_tracker(request: pytest.FixtureRequest):
    """Linux-only diagnostic: records fd count after each test.

    Used to test the CI-hang hypothesis (fd exhaustion from leaked SQLite
    connections / subprocesses). Silent unless running on Linux under CI
    (``CI`` env var set), so it stays cheap everywhere else.
    """
    if not (sys.platform.startswith("linux") and os.environ.get("CI")):
        yield
        return
    yield
    fds = _linux_fd_count()
    if fds >= 0:
        _fd_growth.append((request.node.nodeid, fds))
        if len(_fd_growth) % 50 == 0:
            children = _linux_child_count()
            print(
                f"[fd-growth] after {len(_fd_growth)} tests: fds={fds} children={children}",
                file=sys.stderr,
            )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not (sys.platform.startswith("linux") and os.environ.get("CI")):
        return
    if not _fd_growth:
        return
    out_path = Path("data") / "fd_growth.txt"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"total tests recorded: {len(_fd_growth)}"]
        prevs = [0] + [f for _, f in _fd_growth[:-1]]
        deltas = [
            (nodeid, fds, fds - prev)
            for prev, (nodeid, fds) in zip(prevs, _fd_growth, strict=True)
        ]
        top = sorted(deltas, key=lambda t: t[2], reverse=True)[:25]
        lines.append("top 25 tests by fd growth (nodeid, fds_after, delta):")
        for nodeid, fds, delta in top:
            lines.append(f"  {delta:+5d}  fds={fds:5d}  {nodeid}")
        lines.append("running total every 50 tests:")
        for i in range(49, len(_fd_growth), 50):
            nodeid, fds = _fd_growth[i]
            lines.append(f"  test #{i + 1}: fds={fds}  ({nodeid})")
        out_path.write_text("\n".join(lines) + "\n")
    except OSError:
        pass


@pytest.fixture(scope="session")
def fixture_zim(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ~50-article school-topics ZIM with full-text indexing and redirects."""
    zim_dir = tmp_path_factory.mktemp("fixture_zim")
    return zim_fixtures.build_zim(zim_dir / "fixture_en_school.zim", indexing=True)


@pytest.fixture(scope="session")
def truncated_zim(tmp_path_factory: pytest.TempPathFactory, fixture_zim: Path) -> Path:
    """A copy of ``fixture_zim`` cut to 60% of its bytes."""
    zim_dir = tmp_path_factory.mktemp("truncated_zim")
    return zim_fixtures.build_truncated_zim(fixture_zim, zim_dir / "truncated.zim")


@pytest.fixture(scope="session")
def not_a_zim(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A plain text file renamed to end in ``.zim``."""
    zim_dir = tmp_path_factory.mktemp("not_a_zim")
    return zim_fixtures.build_not_a_zim(zim_dir / "not_a.zim")


@pytest.fixture(scope="session")
def noindex_zim(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small ZIM (~5 articles) built with full-text indexing disabled."""
    zim_dir = tmp_path_factory.mktemp("noindex_zim")
    return zim_fixtures.build_zim(
        zim_dir / "noindex.zim", indexing=False, include_rich=False, n_simple=5
    )

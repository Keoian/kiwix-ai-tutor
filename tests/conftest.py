"""Session-scoped ZIM fixtures shared by WP-B1/WP-B2 tests.

The fixtures are built once per test session with ``libzim.writer.Creator``
(see ``tests/zim_fixtures.py``) so tests exercise real libzim binaries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import zim_fixtures


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

"""Integration check: validate every real archive in config/archives.dev.toml.

Not run in CI (see pyproject.toml's ``integration`` marker). Skips any
archive whose file does not exist on this machine; opens (but never runs
``Archive.check()`` on) the archives that do exist, since D:\\kiwix here is a
slow HDD and a full integrity check is far too slow for a routine run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.retrieval.registry import load_registry
from tutor.retrieval.zim.archive import ArchiveState

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_validate_all_real_archives() -> None:
    registry = load_registry(REPO_ROOT / "config" / "archives.dev.toml")
    statuses = registry.validate_all()

    checked = 0
    for entry in registry.archives:
        status = statuses[entry.id]
        if status.state is ArchiveState.MISSING:
            continue
        checked += 1
        # validate_archive must never raise, and every present file must
        # resolve to *some* well-defined state (asserted implicitly by
        # reaching this point); we do not require every real-world archive
        # on this machine to be VALID, since a corrupt/partial download is
        # exactly the sort of thing this registry needs to detect and
        # report rather than crash on.
        assert isinstance(status.state, ArchiveState)
    assert checked > 0

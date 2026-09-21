"""Integration check (retrieval v16): "modifiers are not topics" against
the real simplewiki archive.

Not run in CI (see pyproject.toml's ``integration`` marker; follows the
skip-if-missing pattern of ``tests/test_registry_real_archives.py``). Uses
``config/archives.simplewiki_only.toml`` -- the one archive the bug report
(see docs/retrieval_baseline.md "v16") was measured against -- and skips
outright if that file is not present on this machine.

Six real questions measured to surface junk-title passages before the v16
fix (see the task brief / docs/retrieval_baseline.md "v16"):
    - "Is DNA the longest molecule?" -> "The Longest Ride" (a movie)
    - "What's the biggest animal?" -> "The Biggest Loser", "World's
      Biggest Coffee Morning"
    - "What's the fastest bird?" -> "Fastest lap" (motorsport)
    - "How long is DNA?" -> "Long Island (disambiguation)"
    - "What's the tallest mountain?" -> "List of tallest buildings in
      Australia" ranked above "Mountain"
    - "What's the longest river?" -> ZERO passages (status empty)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tutor.retrieval.registry import load_registry
from tutor.retrieval.research import ResearchEngine
from tutor.retrieval.snapshots import SnapshotStore

REPO_ROOT = Path(__file__).resolve().parents[1]
_SIMPLEWIKI_PATH = Path(r"C:\kiwix\wikipedia_en_simple_all_maxi_2026-05.zim")

_JUNK_TITLES = (
    "The Longest Ride",
    "The Biggest Loser",
    "World's Biggest Coffee Morning",
    "Fastest lap",
    "Long Island (disambiguation)",
    "List of tallest buildings in Australia",
)


@pytest.mark.integration
def test_modifier_questions_never_return_junk_titles(tmp_path) -> None:
    if not _SIMPLEWIKI_PATH.exists():
        pytest.skip(f"simplewiki archive not present at {_SIMPLEWIKI_PATH}")

    registry = load_registry(REPO_ROOT / "config" / "archives.simplewiki_only.toml")
    store = SnapshotStore(tmp_path / "snapshots.sqlite3")
    try:
        engine = ResearchEngine(
            registry,
            snapshot_store=store,
            cache_dir=tmp_path / "cache",
        )

        questions = [
            "Is DNA the longest molecule?",
            "What's the biggest animal?",
            "What's the fastest bird?",
            "How long is DNA?",
            "What's the tallest mountain?",
            "What's the longest river?",
        ]
        for question in questions:
            response = engine.research(question)
            titles = [p.title for p in response.passages]
            for junk in _JUNK_TITLES:
                assert junk not in titles, f"{question!r} returned junk title {junk!r}: {titles}"

        # "What's the longest river?" specifically must not come back
        # empty (Baseline bug: the modifier "longest" was being required
        # for coverage, which a real river article's text rarely echoes
        # verbatim).
        river_response = engine.research("What's the longest river?")
        assert river_response.passages, (
            f"'What's the longest river?' returned no passages "
            f"(status={river_response.status!r})"
        )
    finally:
        store.close()

"""RED tests for ``tutor.retrieval.zim.search`` (WP-B3).

In-process thin wrappers over libzim.search / libzim.suggestion. Uses the
session-scoped fixture ZIM archives from tests/conftest.py + zim_fixtures.py.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from libzim.reader import Archive

from tutor.retrieval.zim.resolve import EntryNotFound
from tutor.retrieval.zim.search import (
    FetchedEntry,
    NoFulltextIndex,
    SearchHit,
    estimated_matches,
    fetch_entry,
    search_fulltext,
    search_titles,
)


def test_search_hit_is_frozen_dataclass_with_expected_fields() -> None:
    hit = SearchHit(
        path="pythagorean_theorem",
        title="Pythagorean theorem",
        rank=0,
        snippet="",
        source="fulltext",
    )
    assert hit.path == "pythagorean_theorem"
    assert hit.rank == 0
    with pytest.raises(FrozenInstanceError):
        hit.rank = 1  # type: ignore[misc]


def test_search_fulltext_returns_hits_for_known_topic(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    hits = search_fulltext(archive, "Pythagorean theorem", limit=20)
    assert len(hits) > 0
    assert all(isinstance(h, SearchHit) for h in hits)
    assert any(h.path == "pythagorean_theorem" for h in hits)
    assert all(h.source == "fulltext" for h in hits)


def test_search_fulltext_respects_limit(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    hits = search_fulltext(archive, "the", limit=3)
    assert len(hits) <= 3


def test_search_fulltext_empty_query_returns_empty_list(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    assert search_fulltext(archive, "", limit=20) == []


def test_search_fulltext_whitespace_query_returns_empty_list(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    assert search_fulltext(archive, "   ", limit=20) == []


def test_search_fulltext_ranks_are_zero_based_and_ordered(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    hits = search_fulltext(archive, "Pythagoras", limit=20)
    assert len(hits) > 0
    ranks = [h.rank for h in hits]
    assert ranks == sorted(ranks)
    assert ranks[0] == 0


def test_search_fulltext_deterministic_across_repeated_calls(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    first = search_fulltext(archive, "Pythagorean theorem", limit=20)
    second = search_fulltext(archive, "Pythagorean theorem", limit=20)
    assert first == second


def test_search_fulltext_resolves_redirects_to_final_path_and_dedupes(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    hits = search_fulltext(archive, "Pythagorean theorem", limit=20)
    paths = [h.path for h in hits]
    # redirect_a / redirect_b must never appear as raw paths in results
    assert "redirect_a" not in paths
    assert "redirect_b" not in paths
    # no duplicates by final path
    assert len(paths) == len(set(paths))


def test_search_fulltext_raises_no_fulltext_index_on_noindex_zim(noindex_zim: Path) -> None:
    archive = Archive(str(noindex_zim))
    with pytest.raises(NoFulltextIndex):
        search_fulltext(archive, "photosynthesis", limit=10)


def test_estimated_matches_nonzero_for_known_topic(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    assert estimated_matches(archive, "Pythagoras") > 0


def test_estimated_matches_zero_for_offcorpus_term(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    assert estimated_matches(archive, "zzqxwfnorbleasdkjqwe") == 0


def test_estimated_matches_zero_for_blank_query(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    assert estimated_matches(archive, "   ") == 0


def test_estimated_matches_rarer_term_has_fewer_matches(fixture_zim: Path) -> None:
    # "Erdős" appears in only 2 of the fixture's ~55 articles; "school"
    # appears in ~50 of them (every generated simple-topic page).
    archive = Archive(str(fixture_zim))
    assert estimated_matches(archive, "Erdős") < estimated_matches(archive, "school")


def test_estimated_matches_zero_on_noindex_zim(noindex_zim: Path) -> None:
    archive = Archive(str(noindex_zim))
    assert estimated_matches(archive, "photosynthesis") == 0


def test_search_titles_returns_hits_for_known_topic(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    hits = search_titles(archive, "Pythagorean", limit=10)
    assert len(hits) > 0
    assert all(h.source == "title" for h in hits)


def test_search_titles_respects_limit(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    hits = search_titles(archive, "the", limit=2)
    assert len(hits) <= 2


def test_search_titles_empty_query_returns_empty_list(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    assert search_titles(archive, "", limit=10) == []


def test_search_titles_works_on_noindex_zim_without_raising(noindex_zim: Path) -> None:
    archive = Archive(str(noindex_zim))
    hits = search_titles(archive, "Photosynthesis", limit=10)
    assert len(hits) >= 0


def test_search_titles_deterministic_across_repeated_calls(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    first = search_titles(archive, "Pythagorean", limit=10)
    second = search_titles(archive, "Pythagorean", limit=10)
    assert first == second


def test_fetch_entry_returns_expected_fields(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    entry = fetch_entry(archive, "pythagorean_theorem")
    assert isinstance(entry, FetchedEntry)
    assert entry.path == "pythagorean_theorem"
    assert entry.mimetype == "text/html"
    assert entry.hops == ()
    assert "Pythagorean" in entry.title


def test_fetch_entry_unicode_roundtrip_superscripts(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    entry = fetch_entry(archive, "pythagorean_theorem")
    assert "a²+b²=c²" in entry.html


def test_fetch_entry_unicode_roundtrip_erdos(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    entry = fetch_entry(archive, "history_of_mathematics")
    assert "Erdős" in entry.html


def test_fetch_entry_resolves_redirect_and_reports_hops(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    entry = fetch_entry(archive, "redirect_a")
    assert entry.path == "pythagorean_theorem"
    assert entry.hops == ("redirect_a",)


def test_fetch_entry_missing_path_raises_entry_not_found(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    with pytest.raises(EntryNotFound):
        fetch_entry(archive, "this_path_does_not_exist")

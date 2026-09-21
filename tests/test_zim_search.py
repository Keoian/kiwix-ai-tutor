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


def test_search_fulltext_snippet_top_n_none_is_byte_identical_default(
    fixture_zim: Path,
) -> None:
    """Baseline v9: default (snippet_top_n=None) must be unchanged."""
    archive = Archive(str(fixture_zim))
    default_hits = search_fulltext(archive, "the", limit=10)
    explicit_none_hits = search_fulltext(archive, "the", limit=10, snippet_top_n=None)
    assert [h.snippet for h in default_hits] == [h.snippet for h in explicit_none_hits]
    assert [h.path for h in default_hits] == [h.path for h in explicit_none_hits]


def test_search_fulltext_snippet_top_n_limits_real_snippets(fixture_zim: Path) -> None:
    """Baseline v9 candidate B: only the first N kept hits (by rank) get a
    non-empty snippet; the rest score with ``""`` -- a ranking-changing
    knob, so it must be explicitly requested."""
    archive = Archive(str(fixture_zim))
    hits = search_fulltext(archive, "the", limit=10, snippet_top_n=2)
    assert len(hits) > 2
    non_empty = [h for h in hits if h.snippet]
    # Real prose (unlike titles) usually yields a non-empty snippet for the
    # top hits; assert the cutoff shape rather than exact article content.
    assert len(non_empty) <= 2
    for hit in hits[2:]:
        assert hit.snippet == ""


def test_search_fulltext_text_cache_is_content_identical_when_enabled(
    fixture_zim: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Baseline v9 candidate A: enabling the text cache must never change
    any snippet string -- only whether extraction is repeated."""
    import tutor.retrieval.zim.search as search_mod

    archive = Archive(str(fixture_zim))
    baseline_hits = search_fulltext(archive, "Pythagorean theorem", limit=10)

    monkeypatch.setattr(search_mod, "_TEXT_CACHE_ENABLED", True)
    search_mod._default_text_cache.clear()
    cached_hits = search_fulltext(archive, "Pythagorean theorem", limit=10)
    # Run again to exercise an actual cache hit for every path.
    cached_hits_again = search_fulltext(archive, "Pythagorean theorem", limit=10)

    assert [h.snippet for h in baseline_hits] == [h.snippet for h in cached_hits]
    assert [h.snippet for h in cached_hits] == [h.snippet for h in cached_hits_again]
    assert [h.path for h in baseline_hits] == [h.path for h in cached_hits]

"""Validates the fixture-building infrastructure itself.

These tests must PASS now: they prove the fixtures in ``tests/conftest.py``
/ ``tests/zim_fixtures.py`` are usable ZIM archives before any tutor code
depends on them.
"""

from __future__ import annotations

from pathlib import Path

from libzim.reader import Archive
from libzim.search import Query, Searcher


def test_fixture_zim_opens(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    assert archive.has_main_entry
    assert archive.article_count > 40


def test_fixture_zim_has_fulltext_index(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    assert archive.has_fulltext_index


def test_fixture_zim_searcher_finds_a_word(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    searcher = Searcher(archive)
    search = searcher.search(Query().set_query("Pythagoras"))
    results = list(search.getResults(0, 10))
    assert results, "expected at least one search hit for 'Pythagoras'"


def test_fixture_zim_redirects_resolve(fixture_zim: Path) -> None:
    archive = Archive(str(fixture_zim))
    a = archive.get_entry_by_path("redirect_a")
    assert a.is_redirect
    resolved_a = a.get_redirect_entry()
    assert resolved_a.path == "pythagorean_theorem"

    b = archive.get_entry_by_path("redirect_b")
    assert b.is_redirect
    hop1 = b.get_redirect_entry()
    assert hop1.path == "redirect_a"
    hop2 = hop1.get_redirect_entry()
    assert hop2.path == "pythagorean_theorem"


def test_truncated_zim_is_smaller(fixture_zim: Path, truncated_zim: Path) -> None:
    assert truncated_zim.stat().st_size < fixture_zim.stat().st_size
    assert truncated_zim.stat().st_size > 0


def test_not_a_zim_is_plain_text(not_a_zim: Path) -> None:
    text = not_a_zim.read_text(encoding="utf-8")
    assert "not a ZIM archive" in text


def test_noindex_zim_has_no_fulltext_index(noindex_zim: Path) -> None:
    archive = Archive(str(noindex_zim))
    assert not archive.has_fulltext_index
    assert archive.article_count >= 1

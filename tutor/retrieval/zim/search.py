"""In-process thin wrappers over libzim's full-text search, title suggestion,
and entry fetch, used both directly and by :mod:`tutor.retrieval.zim.worker`.

Search results returned by libzim already carry final (non-redirect) entry
paths, but result paths are re-resolved through
:func:`tutor.retrieval.zim.resolve.resolve_entry` anyway so a hit is never
reported at a redirect path and duplicate final paths are collapsed to one
hit each, keeping the ranking's relative order.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup
from libzim.search import Query, Searcher
from libzim.suggestion import SuggestionSearcher

from tutor.retrieval.zim.resolve import resolve_entry

_SNIPPET_MAX_CHARS = 200
_SNIPPET_CONTEXT_CHARS = 80


class NoFulltextIndex(Exception):
    """Raised by :func:`search_fulltext` when the archive has no Xapian index."""


@dataclass(frozen=True)
class SearchHit:
    """One ranked search result, already resolved to its final entry path."""

    path: str
    title: str
    rank: int
    snippet: str
    source: str


@dataclass(frozen=True)
class FetchedEntry:
    """Plain, picklable data for a fetched (and redirect-resolved) entry."""

    path: str
    title: str
    mimetype: str
    html: str
    hops: tuple[str, ...]


def _is_blank(query: str) -> bool:
    return not query or not query.strip()


def _build_snippet(html: str, query: str) -> str:
    """Cheap snippet: plain text around the first query-term match.

    libzim's Python binding does not expose Xapian snippets, so this is
    constructed from the fetched entry's own HTML via bs4 text extraction.
    """
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    if not text:
        return ""
    terms = [t for t in query.split() if t]
    lower_text = text.lower()
    match_index = -1
    for term in terms:
        idx = lower_text.find(term.lower())
        if idx != -1:
            match_index = idx
            break
    if match_index == -1:
        snippet = text[:_SNIPPET_MAX_CHARS]
    else:
        start = max(0, match_index - _SNIPPET_CONTEXT_CHARS)
        end = min(len(text), start + _SNIPPET_MAX_CHARS)
        snippet = text[start:end]
    if len(snippet) > _SNIPPET_MAX_CHARS:
        snippet = snippet[:_SNIPPET_MAX_CHARS]
    return snippet.strip()


def _resolve_hits(
    archive: Any,
    raw_paths: list[str],
    *,
    limit: int,
    source: str,
    query: str,
    with_snippet: bool,
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        if len(hits) >= limit:
            break
        try:
            resolved = resolve_entry(archive, raw_path)
        except Exception:  # noqa: BLE001 - a bad result path is simply skipped
            continue
        final_path = resolved.final_path
        if final_path in seen:
            continue
        seen.add(final_path)
        entry = resolved.entry
        snippet = ""
        if with_snippet:
            try:
                item = entry.get_item()
                content = bytes(item.content).decode("utf-8", errors="replace")
                snippet = _build_snippet(content, query)
            except Exception:  # noqa: BLE001 - snippet is best-effort
                snippet = ""
        hits.append(
            SearchHit(
                path=final_path,
                title=entry.title,
                rank=len(hits),
                snippet=snippet,
                source=source,
            )
        )
    return hits


def search_fulltext(archive: Any, query: str, *, limit: int = 20) -> list[SearchHit]:
    """Full-text (Xapian) search over ``archive``.

    Raises :class:`NoFulltextIndex` if the archive has no full-text index.
    Returns ``[]`` for an empty or whitespace-only ``query`` without
    touching the index at all.
    """
    if _is_blank(query):
        return []
    if not bool(archive.has_fulltext_index):
        raise NoFulltextIndex(
            "Archive has no full-text (Xapian) index; use search_titles instead."
        )
    searcher = Searcher(archive)
    search = searcher.search(Query().set_query(query))
    # Over-fetch to survive redirect-collapsing dedup while staying cheap.
    fetch_count = max(limit * 3, limit + 10)
    raw_paths = list(search.getResults(0, fetch_count))
    return _resolve_hits(
        archive, raw_paths, limit=limit, source="fulltext", query=query, with_snippet=True
    )


def search_titles(archive: Any, query: str, *, limit: int = 10) -> list[SearchHit]:
    """Title-suggestion search over ``archive`` (works without a fulltext index)."""
    if _is_blank(query):
        return []
    searcher = SuggestionSearcher(archive)
    suggestion_search = searcher.suggest(query)
    fetch_count = max(limit * 3, limit + 10)
    raw_paths = list(suggestion_search.getResults(0, fetch_count))
    return _resolve_hits(
        archive, raw_paths, limit=limit, source="title", query=query, with_snippet=False
    )


def fetch_entry(archive: Any, path: str) -> FetchedEntry:
    """Fetch ``path``'s entry (following redirects) as plain, picklable data."""
    resolved = resolve_entry(archive, path)
    entry = resolved.entry
    item = entry.get_item()
    html = _html.unescape(bytes(item.content).decode("utf-8", errors="replace"))
    return FetchedEntry(
        path=resolved.final_path,
        title=entry.title,
        mimetype=item.mimetype,
        html=html,
        hops=resolved.hops,
    )

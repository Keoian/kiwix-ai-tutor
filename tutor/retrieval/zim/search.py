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
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from bs4 import BeautifulSoup
from libzim.search import Query, Searcher
from libzim.suggestion import SuggestionSearcher

from tutor.retrieval.zim.content import HTML_PARSER
from tutor.retrieval.zim.resolve import resolve_entry

_SNIPPET_MAX_CHARS = 200
_SNIPPET_CONTEXT_CHARS = 80

# Baseline v9 (docs/retrieval_baseline.md): the ~20 ms/hit snippet cost is
# almost entirely fetch_entry + bs4 text extraction inside _resolve_hits,
# repeated per hit even when the same article path is fetched for a
# different query within (or across) requests. Candidate A: cache the
# extracted plain text (not the final snippet -- the snippet's match
# position depends on the query) keyed by (archive identity, path), so a
# repeat path skips fetch+parse and only re-runs the cheap substring scan.
# Default ON as of Baseline v9: measured byte-identical output (passages
# and snippet text) on all 42 tuning questions, mean latency 1.32s -> 1.03s
# (data/perq_v9_default_run2.json vs data/perq_v9_A_run2.json). Set this env
# var to "0" to force it off (e.g. to reproduce pre-v9 timings exactly).
_TEXT_CACHE_ENABLED = os.environ.get("TUTOR_RETRIEVAL_SNIPPET_TEXT_CACHE", "1") != "0"

# Baseline v10 (docs/retrieval_baseline.md): profiling ~200 real miss-path
# hits found bs4 DOM construction (``BeautifulSoup(html, "html.parser")``)
# is 78% of per-hit miss cost (~17.5 ms/hit); fetch (get_item + bytes()) is
# 20% (~4.5 ms/hit); decode and get_text are each under 2%. No faster parser
# backend is installed (checked ``pip list``: no lxml, selectolax, or
# html5lib) -- adding one was not done here per the task brief; it would
# need to be measured and proposed separately. Given that, this baseline's
# two safe, ranking-preserving wins are both memoisation, not parsing:
#
# 1. bound the existing text cache by bytes instead of a fixed entry count,
#    so more distinct articles stay resident within an explicit, small
#    memory budget (see ``_CACHE_MAX_BYTES`` below) instead of an arbitrary
#    256-entry cap.
# 2. share the fetched/decoded HTML between this module's snippet path and
#    ``fetch_entry`` (used by the passage-extraction pipeline, see
#    ``research.py`` call site 4 and ``tutor.retrieval.zim.bundle``): the
#    articles ``fetch_entry`` is called for (``top_paths``) are drawn from
#    the same search hits that already had a snippet computed, so caching
#    the *raw decoded HTML* (not the final text -- ``fetch_entry`` and the
#    snippet path apply different post-processing to it) lets a repeat path
#    skip get_item()+decode (~20% of miss cost) even though each caller
#    still parses independently (their downstream algorithms differ:
#    ``build_bundle`` renders a structured, furniture-stripped document;
#    the snippet path just wants ``get_text()``).
#
# Memory bound: the spec (docs/plan/offline_tutor_spec_v0.3.md, "16 GB RAM"
# row) budgets the retrieval worker at <= 1 GiB total. Both caches below are
# capped at 20 MiB of cached string content each (~40 MiB combined) --
# roughly 2% of that budget, measured in Python string length as a cheap
# proxy for UTF-8 byte size (an undercount for non-ASCII text, which is rare
# in this corpus; never an overcount that would blow the bound the other
# way in the cases that matter).
_CACHE_MAX_BYTES = 20 * 1024 * 1024


class _TextCache:
    """Per-process LRU cache from (archive identity, path) -> a string,
    bounded by total cached string length rather than entry count (Baseline
    v10) so cache capacity is an explicit, small memory budget."""

    def __init__(self, max_bytes: int = _CACHE_MAX_BYTES) -> None:
        self.max_bytes = max_bytes
        self._data: OrderedDict[tuple[int, str], str] = OrderedDict()
        self._total_bytes = 0

    def get_or_compute(self, archive: Any, path: str, compute: Callable[[], str]) -> str:
        key = (id(archive), path)
        cached = self._data.get(key)
        if cached is not None:
            self._data.move_to_end(key)
            return cached
        value = compute()
        self._data[key] = value
        self._data.move_to_end(key)
        self._total_bytes += len(value)
        while self._total_bytes > self.max_bytes and len(self._data) > 1:
            _, evicted = self._data.popitem(last=False)
            self._total_bytes -= len(evicted)
        return value

    def clear(self) -> None:
        self._data.clear()
        self._total_bytes = 0


_default_text_cache = _TextCache()
# Baseline v10: raw decoded HTML (pre-unescape), shared between the snippet
# path (_resolve_hits) and fetch_entry -- see the note above.
_default_html_cache = _TextCache()


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


def _extract_text(html: str) -> str:
    """bs4 plain-text extraction -- the expensive, query-independent half
    of snippet building (Baseline v9). Baseline v12: parser backend is
    ``tutor.retrieval.zim.content.HTML_PARSER`` (lxml when available, else
    ``html.parser``) -- proven byte-identical for this call site over
    10,869 real articles."""
    return BeautifulSoup(html, HTML_PARSER).get_text(separator=" ", strip=True)


def _snippet_from_text(text: str, query: str) -> str:
    """Cheap, query-dependent half: pick the window around the first term
    match. Split out of ``_build_snippet`` so the extracted text can be
    cached and reused across queries that hit the same article path."""
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


def _build_snippet(html: str, query: str) -> str:
    """Cheap snippet: plain text around the first query-term match.

    libzim's Python binding does not expose Xapian snippets, so this is
    constructed from the fetched entry's own HTML via bs4 text extraction.
    """
    return _snippet_from_text(_extract_text(html), query)


def _resolve_hits(
    archive: Any,
    raw_paths: list[str],
    *,
    limit: int,
    source: str,
    query: str,
    with_snippet: bool,
    snippet_top_n: int | None = None,
) -> list[SearchHit]:
    """
    ``snippet_top_n`` (Baseline v9, candidate B, ranking-changing, OFF unless
    passed): when set, only the first ``snippet_top_n`` kept hits (by Xapian
    rank -- i.e. ``raw_paths`` order, the same order this loop already keeps
    hits in) get a real snippet; hits beyond it get ``""``. Changing this
    changes ranking, since ``_score_articles`` reads ``.snippet`` of every
    hit -- callers must opt in explicitly.
    """
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
        want_snippet = with_snippet and (
            snippet_top_n is None or len(hits) < snippet_top_n
        )
        if want_snippet:
            try:
                if _TEXT_CACHE_ENABLED:

                    def _fetch_raw_html(entry: Any = entry) -> str:
                        return bytes(entry.get_item().content).decode(
                            "utf-8", errors="replace"
                        )

                    text = _default_text_cache.get_or_compute(
                        archive,
                        final_path,
                        lambda entry=entry: _extract_text(
                            _default_html_cache.get_or_compute(
                                archive, final_path, _fetch_raw_html
                            )
                        ),
                    )
                else:
                    item = entry.get_item()
                    content = bytes(item.content).decode("utf-8", errors="replace")
                    text = _extract_text(content)
                snippet = _snippet_from_text(text, query)
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


def search_fulltext(
    archive: Any, query: str, *, limit: int = 20, snippet_top_n: int | None = None
) -> list[SearchHit]:
    """Full-text (Xapian) search over ``archive``.

    Raises :class:`NoFulltextIndex` if the archive has no full-text index.
    Returns ``[]`` for an empty or whitespace-only ``query`` without
    touching the index at all. ``snippet_top_n`` is Baseline v9's
    ranking-changing candidate B -- see :func:`_resolve_hits`; leaving it
    ``None`` keeps today's byte-identical behavior.
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
        archive,
        raw_paths,
        limit=limit,
        source="fulltext",
        query=query,
        with_snippet=True,
        snippet_top_n=snippet_top_n,
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


def estimated_matches(archive: Any, term: str) -> int:
    """Xapian's own estimated total-hit count for ``term`` in ``archive``.

    Used as a real corpus-IDF specificity signal for candidate generation
    (rarer terms like "helium" outrank generic ones like "output" or
    "boiling") without needing an extra per-term search+resolve pass just
    to learn a count. Returns ``0`` for a blank query or an archive with no
    full-text index (the same "no signal" value a truly absent term would
    give), never raises.
    """
    if _is_blank(term):
        return 0
    if not bool(archive.has_fulltext_index):
        return 0
    try:
        searcher = Searcher(archive)
        search = searcher.search(Query().set_query(term))
        return int(search.getEstimatedMatches())
    except Exception:  # noqa: BLE001 - best-effort rarity signal
        return 0


def fetch_entry(archive: Any, path: str) -> FetchedEntry:
    """Fetch ``path``'s entry (following redirects) as plain, picklable data."""
    resolved = resolve_entry(archive, path)
    entry = resolved.entry
    item = entry.get_item()
    if _TEXT_CACHE_ENABLED:
        raw_html = _default_html_cache.get_or_compute(
            archive,
            resolved.final_path,
            lambda item=item: bytes(item.content).decode("utf-8", errors="replace"),
        )
    else:
        raw_html = bytes(item.content).decode("utf-8", errors="replace")
    html = _html.unescape(raw_html)
    return FetchedEntry(
        path=resolved.final_path,
        title=entry.title,
        mimetype=item.mimetype,
        html=html,
        hops=resolved.hops,
    )

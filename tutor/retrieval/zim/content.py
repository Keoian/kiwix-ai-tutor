"""HTML -> plain-text rendering for ZIM articles (WP-B4).

Adapted from openzim-mcp
Upstream path: openzim_mcp/content_processor.py
Tag: v3.3.4
Commit: 9358db06f205bb0b95cc938c68405851a0e205a8
MIT License
Copyright (c) 2025-2026 Cameron Rye

What was changed: only the *idea* of ``select_main_content`` (prefer
``<article>``/``<main>``/``[role=main]`` when exactly one match carries
text, else fall back to the whole document) was kept; its landmark-priority
list and single-match rule are reproduced here in a few lines. Everything
else in the donor's ``content_processor.py`` -- html2text-based Markdown
rendering, the heading-locator regex cascade (``_locate_heading_text`` /
``_match_decorated_heading_line`` / ``_strip_md_inline_decorations``) that
re-finds headings inside rendered Markdown, search-snippet highlighting,
furniture-heading stripping, and the whole caching/config layer -- was
dropped rather than ported. The donor locates section boundaries by
rendering to Markdown-ish text and then regex-searching for each heading's
text in it, which cannot *guarantee* exact ``text[start:end]`` offsets (a
duplicate heading, or a heading whose text recurs in body prose, can locate
the wrong occurrence). This project's acceptance criterion is exact
code-point offsets, so section boundaries are instead recorded ONLINE, as a
single-pass renderer walks the DOM and emits text -- this file's
``_render_with_headings`` -- rather than being re-derived after the fact.
``extract_infobox`` and ``extract_html_links`` are similarly reimplemented
from scratch, in the "ours" module ``bundle.py``, to the narrower KV-row and
internal-links-only shape this project needs; no donor code for those
functions was copied.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

def _detect_bs_parser() -> str:
    """Baseline v12 (docs/retrieval_baseline.md): lxml was proven to produce
    byte-identical output to ``html.parser`` for this call site's text,
    section, infobox, and link extraction over 10,869 real articles (0
    mismatches), and is faster (p50/p95 both lower). Prefer it, but fall
    back to the stdlib parser when lxml is not importable so the app keeps
    working on a machine where it wasn't installed (e.g. it is dropped from
    the environment, or a constrained platform has no wheel)."""
    try:
        import lxml  # noqa: F401
    except ImportError:
        return "html.parser"
    return "lxml"


# BeautifulSoup parser name for this module's DOM builds (zim-bundle-v2:
# build_bundle's text/sections/infobox/links). See ``_detect_bs_parser``.
HTML_PARSER = _detect_bs_parser()

_HEADING_NAMES = ("h1", "h2", "h3", "h4", "h5", "h6")

# Landmarks tried in priority order; used only when exactly one match exists
# and it carries visible text (ambiguous multi-match pages fall through to
# the whole document). Mirrors the donor's ``_MAIN_CONTENT_SELECTORS``.
_MAIN_CONTENT_SELECTORS = ("article", "main", "[role=main]")

# Tags that are page furniture, never article content, and are dropped
# entirely before rendering.
_FURNITURE_TAGS = frozenset(
    {"script", "style", "nav", "footer", "noscript", "header", "head"}
)

# Block-level tags that emit their own text as one paragraph/line.
_BLOCK_TEXT_TAGS = frozenset({"p", "li", "dt", "dd", "blockquote", "figcaption"})

# Containers: never emit their own text, just recurse into children.
_CONTAINER_TAGS = frozenset(
    {
        "div",
        "section",
        "article",
        "main",
        "aside",
        "span",
        "body",
        "html",
        "ul",
        "ol",
        "dl",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "figure",
        "[document]",
    }
)


def _has_class(tag: Tag, name: str) -> bool:
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    return name in classes


def _strip_furniture(root: Tag) -> None:
    """Remove furniture tags, reference markers, and edit links in place."""
    for tag in list(root.find_all(list(_FURNITURE_TAGS))):
        tag.decompose()
    for sup in list(root.find_all("sup")):
        if _has_class(sup, "reference"):
            sup.decompose()
    for a in list(root.find_all("a")):
        if _has_class(a, "edit"):
            a.decompose()
    for comment in root.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()


def select_main_content(html_or_soup: str | BeautifulSoup) -> Tag:
    """Return the page's main-content subtree, or the whole document.

    Tries ``<article>``, then ``<main>``, then ``[role=main]``; a landmark is
    used only when exactly one match exists and it carries visible text.
    """
    soup = (
        BeautifulSoup(html_or_soup, HTML_PARSER)
        if isinstance(html_or_soup, str)
        else html_or_soup
    )
    for selector in _MAIN_CONTENT_SELECTORS:
        nodes = soup.select(selector)
        if len(nodes) == 1 and nodes[0].get_text(strip=True):
            return nodes[0]
    return soup


def _clean_text(text: str) -> str:
    return " ".join(text.split())


class _Builder:
    """Accumulates rendered text and records heading start offsets online."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self.headings: list[tuple[int, str, int]] = []

    def _length(self) -> int:
        return sum(len(p) for p in self._parts)

    def add_block(self, text: str) -> None:
        text = _clean_text(text)
        if not text:
            return
        if self._parts:
            self._parts.append("\n\n")
        self._parts.append(text)

    def add_heading(self, level: int, text: str) -> None:
        text = _clean_text(text)
        if not text:
            return
        if self._parts:
            self._parts.append("\n\n")
        start = self._length()
        self._parts.append(text)
        self.headings.append((level, text, start))

    def result(self) -> str:
        return "".join(self._parts)


def _table_row_text(tr: Tag) -> str:
    cells = [
        _clean_text(c.get_text(" "))
        for c in tr.find_all(["th", "td"], recursive=False)
    ]
    cells = [c for c in cells if c]
    return " | ".join(cells)


def _walk(node: Tag, builder: _Builder) -> None:
    for child in list(node.children):
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in _HEADING_NAMES:
            level = int(name[1])
            builder.add_heading(level, child.get_text(" "))
        elif name == "tr":
            builder.add_block(_table_row_text(child))
        elif name in _BLOCK_TEXT_TAGS:
            builder.add_block(child.get_text(" "))
        elif name in _CONTAINER_TAGS:
            _walk(child, builder)
        else:
            # Unknown tag: treat as a transparent container.
            _walk(child, builder)


def _render_with_headings(root: Tag) -> tuple[str, list[tuple[int, str, int]]]:
    """Single pass over ``root``: emit text, recording heading offsets online.

    Returns ``(text, headings)`` where ``headings`` is
    ``[(level, heading_text, start_offset), ...]`` in document order and
    ``start_offset`` is the code-point offset into ``text`` where that
    heading's own text begins.
    """
    builder = _Builder()
    _walk(root, builder)
    return builder.result(), builder.headings


def render_text(html: str | BeautifulSoup) -> str:
    """Render ``html`` to deterministic plain text.

    Strips ``<script>``/``<style>``/``<nav>``/``<footer>``/``<head>``,
    reference-marker superscripts (``<sup class="reference">``), and
    ``[edit]``-style links (``<a class="edit">``). Keeps headings,
    paragraphs, and list items, each on their own line, with at most one
    blank line between blocks. Preserves Unicode text exactly (no
    normalization beyond whitespace collapsing).
    """
    soup = BeautifulSoup(html, HTML_PARSER) if isinstance(html, str) else html
    root = select_main_content(soup)
    _strip_furniture(root)
    text, _headings = _render_with_headings(root)
    return text


def iter_internal_links(root: Tag) -> Iterable[str]:
    """Yield internal hrefs from ``root`` in first-seen document order.

    "Internal" excludes links with a network scheme (``http:``, ``https:``,
    ``mailto:``, etc.) or a scheme-relative ``//`` prefix; anything else
    (a bare ZIM path, possibly with a ``#fragment`` or ``?query``) counts as
    internal and is yielded once, deduplicated by exact href string.
    """
    seen: set[str] = set()
    for a in root.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href = a["href"]
        if not isinstance(href, str) or not href:
            continue
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href) or href.startswith("//"):
            continue
        if href in seen:
            continue
        seen.add(href)
        yield href

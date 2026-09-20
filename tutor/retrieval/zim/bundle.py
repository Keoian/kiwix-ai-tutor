"""Single-parse ArticleBundle extraction for ZIM entries (WP-B4).

Adapted from openzim-mcp
Upstream path: openzim_mcp/bundle.py
Tag: v3.3.4
Commit: 9358db06f205bb0b95cc938c68405851a0e205a8
MIT License
Copyright (c) 2025-2026 Cameron Rye

What was changed: only the top-level shape of ``extract_entry_bundle`` --
one HTML parse producing one bundle value that downstream code slices
instead of re-parsing -- was kept. The donor's actual mechanism
(``_compute_section_offsets``, ``_locate_heading_text``,
``_match_decorated_heading_line``, ``_strip_md_inline_decorations``) locates
headings by rendering to Markdown-ish text and then regex-searching for each
heading's text back inside it, which cannot guarantee this project's
acceptance criterion -- exact ``text[start:end]`` slicing for every section,
with no possibility of matching the wrong occurrence of a repeated heading.
That whole cascade was dropped; ``tutor.retrieval.zim.content`` instead
records each heading's offset ONLINE during a single-pass render, and
``_build_sections`` below just turns that list of ``(level, text, start)``
triples into tiled, parent-linked ``SectionMeta`` values by simple
arithmetic. The cache layer (``get_or_build_bundle``, ``archive_stat_token``,
the ``bundle:v2h`` key scheme) and every field tied to the donor's compact
Markdown-table rendering (``word_count``, ``char_count``, ``content_type``,
the ``compact`` flag) were dropped as unneeded here. Infobox extraction is a
small from-scratch KV-row scan (no shared code with the donor's
``ContentProcessor.extract_infobox``, which returns a nested TypedDict this
project has no use for); link extraction is
``content.iter_internal_links``, also written from scratch.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from tutor.retrieval.zim.content import (
    HTML_PARSER,
    _render_with_headings,
    _strip_furniture,
    iter_internal_links,
    select_main_content,
)
from tutor.retrieval.zim.models import ArticleBundle, SectionMeta

# Bump whenever a change to this module or ``content.py`` changes what a
# bundle CONTAINS (which sections exist, their offsets, links, or infobox
# rows) for an unchanged archive -- see docs/bundle_and_passages.md. Anything
# keyed on a bundle (e.g. a passage id, see ``hybrid/passages.py``) folds
# this in so a stale value is never served across an extractor change.
#
# v2 (item 4, real-failure fix): the infobox -- previously extracted into
# ``ArticleBundle.infobox`` but never rendered into ``bundle.text`` -- is
# now ALSO appended as one final synthetic "Infobox" section, so
# ``split_passages`` yields a citable "Key: value" passage for it (e.g.
# Helium's boiling point in a table cell now reaches the model; before this
# it was extracted but silently unreachable). This changes ``bundle.text``
# for every article that has an infobox, which changes every downstream
# passage id for that article (see ``hybrid/passages.py``'s id formula,
# which folds in ``extractor_version``) -- see docs/bundle_and_passages.md
# for what that invalidates.
EXTRACTOR_VERSION = "zim-bundle-v2"

_INFOBOX_SELECTORS = ("table.infobox",)
_INFOBOX_HEADING = "Infobox"


def _extract_infobox(root: Tag) -> tuple[tuple[str, str], ...]:
    """Extract the first ``table.infobox`` as ``(label, value)`` pairs.

    Mutates ``root`` to remove the extracted table so it is not also
    rendered as body text. Returns ``()`` when no infobox is present.
    """
    for selector in _INFOBOX_SELECTORS:
        table = root.select_one(selector)
        if table is None:
            continue
        rows: list[tuple[str, str]] = []
        for tr in table.select("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if isinstance(th, Tag) and isinstance(td, Tag):
                label = " ".join(th.get_text(" ").split())
                value = " ".join(td.get_text(" ").split())
                if label and value:
                    rows.append((label, value))
        table.decompose()
        return tuple(rows)
    return ()


def _build_sections(
    text: str, headings: list[tuple[int, str, int]]
) -> tuple[SectionMeta, ...]:
    """Turn ``(level, heading_text, start)`` triples into tiled sections.

    Only headings at level >= 2 open a new section; an ``<h1>`` is the
    article title and is folded into the lead (index 0, ``heading=""``).
    Sections tile the full text in document order: each section's ``end``
    is simply the next section's ``start`` (or ``len(text)`` for the last),
    regardless of heading level -- nesting is expressed only through
    ``parent_index``/``level``, computed via a level stack seeded with the
    lead as an implicit level-1 ancestor.
    """
    sections: list[SectionMeta] = [
        SectionMeta(index=0, heading="", level=1, parent_index=None, start=0, end=0)
    ]
    stack: list[tuple[int, int]] = [(1, 0)]
    for level, heading_text, start in headings:
        if level < 2:
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_index = stack[-1][1] if stack else None
        idx = len(sections)
        sections.append(
            SectionMeta(
                index=idx,
                heading=heading_text,
                level=level,
                parent_index=parent_index,
                start=start,
                end=0,
            )
        )
        stack.append((level, idx))

    tiled: list[SectionMeta] = []
    for i, section in enumerate(sections):
        end = sections[i + 1].start if i + 1 < len(sections) else len(text)
        tiled.append(
            SectionMeta(
                index=section.index,
                heading=section.heading,
                level=section.level,
                parent_index=section.parent_index,
                start=section.start,
                end=end,
            )
        )
    return tuple(tiled)


def build_bundle(html: str, *, path: str, title: str) -> ArticleBundle:
    """Parse ``html`` once and return the resulting :class:`ArticleBundle`.

    Pure: no caching, no I/O beyond the string already in memory.
    """
    soup = BeautifulSoup(html, HTML_PARSER)
    root = select_main_content(soup)
    _strip_furniture(root)
    links = tuple(iter_internal_links(root))
    infobox = _extract_infobox(root)
    text, headings = _render_with_headings(root)
    if infobox:
        kv_lines = "\n".join(f"{label}: {value}" for label, value in infobox)
        heading_start = len(text) + 2  # after the "\n\n" separator below
        text = f"{text}\n\n{_INFOBOX_HEADING}\n{kv_lines}"
        headings = [*headings, (2, _INFOBOX_HEADING, heading_start)]
    sections = _build_sections(text, headings)
    return ArticleBundle(
        path=path,
        title=title,
        text=text,
        sections=sections,
        links=links,
        infobox=infobox,
        extractor_version=EXTRACTOR_VERSION,
    )

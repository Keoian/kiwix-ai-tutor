"""Frozen dataclasses for ZIM article bundles (WP-B4).

Ours: no donor code. Kept separate from ``bundle.py`` so the value types can
be imported without pulling in the bs4-based extraction code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionMeta:
    """One section of an :class:`ArticleBundle`.

    Sections tile ``ArticleBundle.text`` in document order: for consecutive
    sections ``cur.end == nxt.start``, the first section starts at 0, and the
    last ends at ``len(text)``. ``heading`` is ``""`` for the synthetic lead
    section (index 0), which covers everything before the first ``<h2>`` (an
    ``<h1>`` does not start its own section — it is folded into the lead).
    ``parent_index`` points at the nearest enclosing section with a lower
    heading level (``None`` for the lead).
    """

    index: int
    heading: str
    level: int
    parent_index: int | None
    start: int
    end: int


@dataclass(frozen=True)
class ArticleBundle:
    """The single-parse result for one ZIM article entry.

    ``text`` is the rendered plain-text body; ``sections[i].start``/``end``
    are code-point offsets into it (see ``docs/bundle_and_passages.md``).
    ``links`` are deduplicated internal hrefs in first-seen order. ``infobox``
    is a tuple of ``(label, value)`` pairs, empty when the article has none.
    """

    path: str
    title: str
    text: str
    sections: tuple[SectionMeta, ...]
    links: tuple[str, ...]
    infobox: tuple[tuple[str, str], ...]
    extractor_version: str

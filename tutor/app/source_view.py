"""Source viewer support (WP-C3).

Builds a ``SourceView`` from a stored citation :class:`Snapshot`
(``tutor.retrieval.snapshots.Snapshot``-shaped object): the exact cited
sentence span plus bounded, sentence-snapped context before/after, with any
obviously unsafe markup stripped before it ever reaches the UI layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_UNSAFE_TAG_RE = re.compile(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL)
_UNSAFE_TAG_OPEN_RE = re.compile(r"<\s*script[^>]*>", re.IGNORECASE)


def _strip_unsafe(text: str) -> str:
    text = _UNSAFE_TAG_RE.sub(" ", text)
    text = _UNSAFE_TAG_OPEN_RE.sub(" ", text)
    return text


def sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand ``(start, end)`` out to enclosing sentence boundaries.

    Never narrows past the given span, and never crosses the edges of
    ``text``.
    """
    length = len(text)
    start = max(0, min(start, length))
    end = max(start, min(end, length))

    # Expand start backwards to just after the previous sentence-ending
    # punctuation (or the start of the text).
    search_region = text[:start]
    last_end = None
    for m in _SENTENCE_END_RE.finditer(search_region):
        last_end = m.end()
    if last_end is None:
        s = 0
    else:
        # skip whitespace after the punctuation
        s = last_end
        while s < start and text[s].isspace():
            s += 1
    s = min(s, start)

    # Expand end forwards to the next sentence-ending punctuation (inclusive).
    search_start = max(end - 1, 0)
    m = _SENTENCE_END_RE.search(text, search_start)
    if m is None:
        e = length
    else:
        e = m.end()
    e = max(e, end)
    e = min(e, length)

    return s, e


@dataclass(frozen=True)
class SourceView:
    title: str
    path: str
    archive_id: str
    heading_path: tuple[str, ...]
    text: str
    highlight: tuple[int, int]
    context_before: str
    context_after: str
    kind: str | None


def build_source_view(snapshot: Any, *, context_chars: int = 600) -> SourceView:
    raw_text = snapshot.text
    text = _strip_unsafe(raw_text)

    start = max(0, min(snapshot.start, len(text)))
    end = max(start, min(snapshot.end, len(text)))

    before_bound = max(0, start - context_chars)
    after_bound = min(len(text), end + context_chars)

    ctx_start, _ = sentence_span(text, before_bound, before_bound)
    _, ctx_end = sentence_span(text, after_bound, after_bound)
    ctx_start = max(0, min(ctx_start, start))
    ctx_end = max(end, min(ctx_end, len(text)))

    context_before = _strip_unsafe(text[ctx_start:start])
    context_after = _strip_unsafe(text[end:ctx_end])

    kind = getattr(snapshot, "kind", None)

    return SourceView(
        title=snapshot.title,
        path=snapshot.path,
        archive_id=snapshot.archive_id,
        heading_path=tuple(snapshot.heading_path),
        text=text,
        highlight=(start, end),
        context_before=context_before,
        context_after=context_after,
        kind=kind,
    )

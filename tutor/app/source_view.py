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


# Default expansion either side of the cited passage, in characters, before
# snapping outward to sentence boundaries (see the module docstring's task:
# "Show more of the article"). Kept well under the hard response cap so the
# default request never needs clipping.
_DEFAULT_CONTEXT_CHARS = 1500

# Hard cap on the total length of the ``text`` field a single
# ``/context`` response returns, regardless of how large ``before``/``after``
# are asked for -- callers page with repeated requests instead.
_MAX_CONTEXT_CHARS = 8000


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


def build_context_view(
    snapshot: Any,
    article_text: str,
    *,
    before: int = _DEFAULT_CONTEXT_CHARS,
    after: int = _DEFAULT_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Build the ``GET /api/source/{id}/context`` response payload.

    ``article_text`` is the full article re-rendered by the SAME extraction
    that produced ``snapshot.text`` (see
    ``tutor.retrieval.research.ResearchEngine.fetch_article_text``), so the
    common case is ``article_text[snapshot.start:snapshot.end] ==
    snapshot.text`` exactly. If the archive was re-rendered and offsets
    drifted, falls back to :func:`tutor.retrieval.snapshots.locate_in_text`
    to relocate the passage by substring search; if the passage text is
    simply gone, the whole article text (clipped to the cap) is still
    returned with the passage span clamped to an empty range at 0 so the
    UI has something to show rather than erroring.

    The returned window is expanded ``before``/``after`` characters past
    the passage's own span and then snapped outward to sentence boundaries
    (:func:`sentence_span`), then clipped to ``_MAX_CONTEXT_CHARS`` total
    (trimming the far edges evenly) so one request can never return an
    unbounded amount of text.
    """
    text = _strip_unsafe(article_text)
    length = len(text)

    exact = (
        0 <= snapshot.start <= snapshot.end <= length
        and text[snapshot.start : snapshot.end] == snapshot.text
    )
    if exact:
        span = (snapshot.start, snapshot.end)
    else:
        from tutor.retrieval.snapshots import locate_in_text

        span = locate_in_text(snapshot, text)

    if span is None:
        start = end = 0
    else:
        start, end = span

    before = max(0, before)
    after = max(0, after)

    before_bound = max(0, start - before)
    after_bound = min(length, end + after)

    ctx_start, _ = sentence_span(text, before_bound, before_bound)
    _, ctx_end = sentence_span(text, after_bound, after_bound)
    ctx_start = max(0, min(ctx_start, start))
    ctx_end = max(end, min(ctx_end, length))

    # Hard cap: shrink the window (never past the passage itself) rather
    # than ever returning more than _MAX_CONTEXT_CHARS of text.
    window = ctx_end - ctx_start
    if window > _MAX_CONTEXT_CHARS:
        overflow = window - _MAX_CONTEXT_CHARS
        trim_before = min(overflow, max(0, start - ctx_start))
        ctx_start += trim_before
        overflow -= trim_before
        if overflow > 0:
            trim_after = min(overflow, max(0, ctx_end - end))
            ctx_end -= trim_after

    more_before = ctx_start > 0
    more_after = ctx_end < length

    return {
        "title": snapshot.title,
        "passage": {"start": start - ctx_start, "end": end - ctx_start},
        "text": text[ctx_start:ctx_end],
        "text_start": ctx_start,
        "more_before": more_before,
        "more_after": more_after,
    }

"""Split an ArticleBundle into retrievable, citable passages (WP-B5).

Ours: no donor code. See docs/bundle_and_passages.md for the passage id
formula and its relationship to the spec/reuse-plan wording.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from tutor.retrieval.zim.models import ArticleBundle, SectionMeta

# Soft cap on a passage's character span. A section's words are packed into
# passages up to this size so a long section (e.g. an un-subsectioned wall of
# text) still yields several independently-citable passages instead of one
# giant one; a short section still yields exactly one.
_MAX_PASSAGE_CHARS = 400

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Passage:
    """One citable slice of an :class:`ArticleBundle`.

    ``bundle.text[start:end] == text`` always holds, and ``start``/``end``
    never cross a section boundary. ``heading_path`` is the chain of
    (non-lead) heading titles from the article root down to this passage's
    section, e.g. ``("Parent", "Child")``.
    """

    passage_id: str
    archive_id: str
    path: str
    title: str
    section_index: int
    heading_path: tuple[str, ...]
    start: int
    end: int
    text: str


def _heading_path(
    section: SectionMeta, sections_by_index: dict[int, SectionMeta]
) -> tuple[str, ...]:
    path: list[str] = []
    current: SectionMeta | None = section
    while current is not None:
        if current.heading:
            path.append(current.heading)
        current = (
            sections_by_index.get(current.parent_index)
            if current.parent_index is not None
            else None
        )
    return tuple(reversed(path))


def _chunk_word_spans(words: list[re.Match[str]]) -> list[tuple[int, int]]:
    """Group word matches into contiguous ``(start, end)`` spans.

    A span grows until adding the next word would push it past
    ``_MAX_PASSAGE_CHARS`` (measured from the span's own start), then a new
    span begins. Gaps between spans are therefore exactly the whitespace
    between the last word of one and the first word of the next.
    """
    spans: list[tuple[int, int]] = []
    span_start: int | None = None
    span_end = 0
    for match in words:
        if span_start is None:
            span_start, span_end = match.start(), match.end()
        elif match.end() - span_start > _MAX_PASSAGE_CHARS:
            spans.append((span_start, span_end))
            span_start, span_end = match.start(), match.end()
        else:
            span_end = match.end()
    if span_start is not None:
        spans.append((span_start, span_end))
    return spans


def _passage_id(
    *,
    fingerprint_digest: str,
    archive_id: str,
    path: str,
    section_index: int,
    extractor_version: str,
    text: str,
    start: int,
    end: int,
) -> str:
    """32-hex passage id: sha256(fingerprint + archive + path + section
    index + extractor version + sha256(text) + span), truncated to 32 hex.

    See docs/bundle_and_passages.md for why this is the same rule as the
    spec's shorter summary, at a fuller level of detail.
    """
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    payload = "\x1f".join(
        [
            fingerprint_digest,
            archive_id,
            path,
            str(section_index),
            extractor_version,
            text_hash,
            str(start),
            str(end),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def split_passages(
    bundle: ArticleBundle, *, fingerprint_digest: str, archive_id: str
) -> list[Passage]:
    """Split ``bundle`` into passages, never crossing a section boundary.

    A section whose only content is its own heading (no body text) yields
    no passages. Otherwise the section's text (heading included) is packed
    word-by-word into one or more passages of up to ``_MAX_PASSAGE_CHARS``.
    """
    sections_by_index = {s.index: s for s in bundle.sections}
    passages: list[Passage] = []
    for section in bundle.sections:
        section_text = bundle.text[section.start : section.end]
        body_probe = section_text[len(section.heading) :]
        if not body_probe.strip():
            continue
        words = list(_WORD_RE.finditer(section_text))
        if not words:
            continue
        heading_path = _heading_path(section, sections_by_index)
        for span_start, span_end in _chunk_word_spans(words):
            abs_start = section.start + span_start
            abs_end = section.start + span_end
            text = bundle.text[abs_start:abs_end]
            passage_id = _passage_id(
                fingerprint_digest=fingerprint_digest,
                archive_id=archive_id,
                path=bundle.path,
                section_index=section.index,
                extractor_version=bundle.extractor_version,
                text=text,
                start=abs_start,
                end=abs_end,
            )
            passages.append(
                Passage(
                    passage_id=passage_id,
                    archive_id=archive_id,
                    path=bundle.path,
                    title=bundle.title,
                    section_index=section.index,
                    heading_path=heading_path,
                    start=abs_start,
                    end=abs_end,
                    text=text,
                )
            )
    return passages

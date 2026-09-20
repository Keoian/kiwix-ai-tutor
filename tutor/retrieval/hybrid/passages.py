"""Split an ArticleBundle into retrievable, citable passages (WP-B5).

Ours: no donor code. See docs/bundle_and_passages.md for the passage id
formula and its relationship to the spec/reuse-plan wording.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from tutor.retrieval.hybrid.lexical import singularize, tokenize
from tutor.retrieval.zim.bundle import _INFOBOX_HEADING
from tutor.retrieval.zim.models import ArticleBundle, SectionMeta

# Soft cap on a passage's character span. A section's words are packed into
# passages up to this size so a long section (e.g. an un-subsectioned wall of
# text) still yields several independently-citable passages instead of one
# giant one; a short section still yields exactly one.
_MAX_PASSAGE_CHARS = 400

# Baseline v6 ("infobox key facts"): a matching infobox row is short (a
# label/value pair), so this cap is in ROWS, not characters -- 8 rows is
# comfortably under the ~120-token budget mentioned in the task brief for
# any real infobox (see docs/retrieval_baseline.md "Baseline v6").
_KEY_FACT_MAX_LINES = 8

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


def build_key_fact_passages(
    bundle: ArticleBundle,
    own_terms: frozenset[str] | set[str],
    *,
    fingerprint_digest: str,
    archive_id: str,
    max_lines: int = _KEY_FACT_MAX_LINES,
) -> list[Passage]:
    """Compact, citable passages for infobox rows whose KEY shares an own
    content term of ``own_terms`` (Baseline v6 "infobox key facts").

    A long infobox splits into several ~400-char passages under
    :func:`split_passages`; the one line that actually answers a factual
    question (e.g. "Boiling point: 4.222 K ...") can lose the packing
    competition to prose passages even though the fact is present in
    ``bundle.text``. This scans ``bundle.infobox`` directly and returns one
    passage per contiguous run of matching rows, honouring the same
    ``bundle.text[start:end] == text`` offset contract as
    :func:`split_passages` (so callers may simply pack these ahead of the
    normal ranked passages). Rows are considered in infobox order and
    capped at ``max_lines`` total matches before grouping into runs. Returns
    ``[]`` when there is no infobox, no matching row, or (defensively) if
    the reconstructed row offsets do not slice back to the expected text.
    """
    if not bundle.infobox:
        return []
    section = next((s for s in bundle.sections if s.heading == _INFOBOX_HEADING), None)
    if section is None:
        return []
    norm_own = {singularize(t) for t in own_terms}
    if not norm_own:
        return []

    # The infobox section body is "<heading>\n<label>: <value>\n..." (see
    # ``zim.bundle.build_bundle``) -- reconstruct each row's exact span by
    # walking the same layout rather than re-parsing ``bundle.text``.
    body_start = section.start + len(section.heading) + 1
    spans: list[tuple[int, int]] = []
    pos = body_start
    for label, value in bundle.infobox:
        line_text = f"{label}: {value}"
        start, end = pos, pos + len(line_text)
        if bundle.text[start:end] != line_text:
            return []
        spans.append((start, end))
        pos = end + 1

    matched_indices = [
        i
        for i, (label, _value) in enumerate(bundle.infobox)
        if {singularize(t) for t in tokenize(label)} & norm_own
    ][:max_lines]
    if not matched_indices:
        return []

    runs: list[list[int]] = []
    for i in matched_indices:
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])

    passages: list[Passage] = []
    for run in runs:
        start = spans[run[0]][0]
        end = spans[run[-1]][1]
        text = bundle.text[start:end]
        passage_id = _passage_id(
            fingerprint_digest=fingerprint_digest,
            archive_id=archive_id,
            path=bundle.path,
            section_index=section.index,
            extractor_version=bundle.extractor_version,
            text=text,
            start=start,
            end=end,
        )
        passages.append(
            Passage(
                passage_id=passage_id,
                archive_id=archive_id,
                path=bundle.path,
                title=bundle.title,
                section_index=section.index,
                heading_path=(_INFOBOX_HEADING,),
                start=start,
                end=end,
                text=text,
            )
        )
    return passages

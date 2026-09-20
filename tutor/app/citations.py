"""Citation extraction, resolution, and evidence rendering.

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §8 (numbered
``[S#]`` labels, stable for the lesson) and §11 (``[S#]`` labels resolved
by the host and rejected if unresolved); docs/plan/
offline_tutor_implementation_plan.md §0.4 (Q&A-kind passages carry a
"[Q&A]" marker in the rendered evidence text given to the model).

``extract_labels`` finds every individual ``S#`` label referenced in a
piece of text, including grouped citations like ``[S1, S3]``, in order of
first appearance, de-duplicated.

``resolve_citations`` returns one :class:`Citation` per unique label
referenced in the text; a label with no matching passage is flagged
``unresolved=True`` rather than silently dropped.

``render_evidence`` renders a retrieval packet's passages as the evidence
block appended to the prompt, marking Q&A-kind passages with ``[Q&A]``,
with each passage's ``[S#]`` label at the START of its line and a short
citation reminder appended at the END of the block (never inside a
passage, so it is never mistaken for sourced text).

Measured (eval/run_turn_eval.py, docs/citation_experiment.md, live
1-bit-8B model, fixture ZIM, 5 factual questions): the trailing reminder
alone raised the citation rate from 0.20 to 0.60 -- the biggest single
lever of the variants tried (a one-shot example in the system prompt,
a shorter imperative system prompt, and temperature 0.2 all did worse or
no better). It was adopted as the host's default rendering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tutor.retrieval.hybrid.lexical import tokenize

_LABEL_GROUP_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
_LABEL_RE = re.compile(r"S\d+")

# Splits the answer into rough sentences/bullets for the mechanical support
# check below: on sentence terminators followed by whitespace, or on
# newlines (a bullet list item is its own "sentence" for this purpose).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# 2026-09-20 evidence-dump follow-up (docs/citation_experiment.md): a
# passage "resolving" (its label existed in the packet) is not the same as
# it "supporting" what the sentence carrying the label actually claims. A
# sentence quoting its passage back still counts as supported by this rule
# -- catching wholesale evidence dumping is `detect_evidence_dump`'s job,
# not this one's.
_MIN_SHARED_TERMS = 2
_MIN_SHARED_FRACTION = 0.30

# A bullet/line is considered "dumped" when this much of its own content is
# contained in the cited passage's text.
_DUMP_CONTAINMENT_FRACTION = 0.80
_MIN_DUMPED_BULLETS = 3
_MIN_STACKED_LABELS = 4


@dataclass(frozen=True)
class Citation:
    label: str
    passage_id: str | None = None
    title: str | None = None
    path: str | None = None
    span: tuple[int, int] | None = None
    unresolved: bool = False
    supported: bool = False


def extract_labels(text: str) -> list[str]:
    """Return each individual ``S#`` label referenced in ``text``, in
    order of first appearance, de-duplicated. Grouped citations like
    "[S1, S3]" expand to individual labels."""
    seen: dict[str, None] = {}
    for group in _LABEL_GROUP_RE.findall(text):
        for label in _LABEL_RE.findall(group):
            seen.setdefault(label, None)
    return list(seen)


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _sentence_for_label(text: str, label: str) -> str:
    """The sentence/bullet-line of ``text`` that carries ``[label]`` (the
    first occurrence, if the label appears more than once). Falls back to
    the whole text if no sentence boundary could be found containing it,
    which never happens in practice given ``label in text`` is already
    guaranteed by ``extract_labels``."""
    needle = f"S{label[1:]}" if not label.startswith("S") else label
    for sentence in _sentences(text):
        if re.search(rf"\[{re.escape(needle)}(\s*,|\s*\])", sentence) or f"[{needle}]" in sentence:
            return sentence
    return text


def is_supported(sentence: str, passage_text: str) -> bool:
    """Mechanical, deterministic support check: does ``sentence`` share
    enough content terms with ``passage_text`` to plausibly be drawn from
    it? Rule (documented, not tuned on live data): supported if the
    sentence and passage share at least ``_MIN_SHARED_TERMS`` content terms
    (numbers count as terms -- ``tokenize`` keeps digit runs) OR the shared
    terms are at least ``_MIN_SHARED_FRACTION`` of the sentence's own
    content terms. A sentence that merely quotes the passage back still
    counts as supported by this rule (see ``detect_evidence_dump`` for
    catching wholesale copying)."""
    sentence_terms = set(tokenize(sentence))
    if not sentence_terms:
        return False
    passage_terms = set(tokenize(passage_text))
    shared = sentence_terms & passage_terms
    if len(shared) >= _MIN_SHARED_TERMS:
        return True
    return (len(shared) / len(sentence_terms)) >= _MIN_SHARED_FRACTION


def _dump_signal(text: str, packet_passages: list[dict]) -> tuple[bool, set[str]]:
    """Shared evidence-dump detection for ``detect_evidence_dump`` and
    ``resolve_citations``'s support override below. Two mechanical,
    deterministic signals (2026-09-20 evidence-dump follow-up, see
    docs/citation_experiment.md):

    1. Label-stacking: any single sentence/bullet carries
       ``_MIN_STACKED_LABELS`` (4) or more distinct ``[S#]`` labels -- the
       hallmark of a "here's everything" trailer sentence like
       "...[S1][S2]...[S11]". Every label on such a sentence is flagged.
    2. Wholesale copying: at least ``_MIN_DUMPED_BULLETS`` (3) distinct
       cited sentences/bullets each have ``_DUMP_CONTAINMENT_FRACTION``
       (80%) or more of their own content terms contained in the text of
       the passage their label resolves to -- i.e. the "citation" is
       really just the passage copied back out. If that threshold is
       reached, every one of those copied-back labels is flagged.

    Returns ``(is_dump, flagged_labels)``: ``flagged_labels`` is the set of
    labels implicated in the dump (used to override an otherwise-"quotes
    its source" ``supported=True`` back to ``False`` -- copying a passage
    back verbatim as part of a dump is not the same as a legitimate
    citation quoting its source, see ``is_supported``)."""
    by_label = {p["label"]: p for p in packet_passages}

    per_sentence_labels: dict[str, set[str]] = {}
    for sentence in _sentences(text):
        labels_here: set[str] = set()
        for group in _LABEL_GROUP_RE.findall(sentence):
            labels_here.update(_LABEL_RE.findall(group))
        if labels_here:
            per_sentence_labels[sentence] = labels_here

    stacked_labels: set[str] = set()
    for labels_here in per_sentence_labels.values():
        if len(labels_here) >= _MIN_STACKED_LABELS:
            stacked_labels |= labels_here

    copied_labels: set[str] = set()
    for sentence, labels_here in per_sentence_labels.items():
        sentence_terms = set(tokenize(sentence))
        if not sentence_terms:
            continue
        for label in labels_here:
            passage = by_label.get(label)
            if passage is None:
                continue
            passage_terms = set(tokenize(passage.get("text", "")))
            if not passage_terms:
                continue
            contained = sentence_terms & passage_terms
            if (len(contained) / len(sentence_terms)) >= _DUMP_CONTAINMENT_FRACTION:
                copied_labels.add(label)

    is_dump = bool(stacked_labels) or len(copied_labels) >= _MIN_DUMPED_BULLETS
    flagged = set(stacked_labels)
    if len(copied_labels) >= _MIN_DUMPED_BULLETS:
        flagged |= copied_labels
    return is_dump, flagged


def resolve_citations(text: str, packet_passages: list[dict]) -> list[Citation]:
    """Return one Citation per unique label referenced in ``text``."""
    by_label = {p["label"]: p for p in packet_passages}
    _, flagged_labels = _dump_signal(text, packet_passages)
    citations: list[Citation] = []
    for label in extract_labels(text):
        passage = by_label.get(label)
        if passage is None:
            citations.append(Citation(label=label, unresolved=True, supported=False))
            continue
        span = None
        if "start" in passage and "end" in passage:
            span = (passage["start"], passage["end"])
        sentence = _sentence_for_label(text, label)
        supported = is_supported(sentence, passage.get("text", ""))
        if label in flagged_labels:
            supported = False
        citations.append(
            Citation(
                label=label,
                passage_id=passage.get("id"),
                title=passage.get("title"),
                path=passage.get("path"),
                span=span,
                unresolved=False,
                supported=supported,
            )
        )
    return citations


def detect_evidence_dump(
    text: str, citations: list[Citation], packet_passages: list[dict]
) -> bool:
    """Detect an answer that reproduces evidence wholesale rather than
    citing it. See ``_dump_signal`` for the two mechanical, deterministic
    signals used (label-stacking, wholesale copying). ``citations`` is
    accepted for symmetry with the ``resolve_citations`` output callers
    already have on hand but is not itself needed -- the detection works
    directly from ``text`` and ``packet_passages``.

    Never edits or removes anything from ``text``; this is a read-only
    signal for the ``done`` event's ``evidence_dump`` flag."""
    del citations
    is_dump, _ = _dump_signal(text, packet_passages)
    return is_dump


_CITATION_REMINDER = (
    "Cite only a source that actually supports the sentence, like [S1]. "
    "Do not list or copy the sources. If none of them answers the "
    "question, say so."
)


def render_evidence(packet: dict) -> str:
    """Render a retrieval packet's passages as an evidence block, one
    line/block per passage (label at the start), marking Q&A-kind
    passages with "[Q&A]", followed by a short citation reminder line --
    unless the packet has no passages, in which case an empty string is
    returned (no reminder to cite evidence that was never shown)."""
    passages = packet.get("passages", [])
    if not passages:
        return ""
    lines = []
    for passage in passages:
        marker = " [Q&A]" if passage.get("kind") == "qa" else ""
        lines.append(f"[{passage['label']}]{marker} {passage.get('text', '')}")
    lines.append(_CITATION_REMINDER)
    return "\n".join(lines)

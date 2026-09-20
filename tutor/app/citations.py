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
block appended to the prompt, marking Q&A-kind passages with ``[Q&A]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LABEL_GROUP_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
_LABEL_RE = re.compile(r"S\d+")


@dataclass(frozen=True)
class Citation:
    label: str
    passage_id: str | None = None
    title: str | None = None
    path: str | None = None
    span: tuple[int, int] | None = None
    unresolved: bool = False


def extract_labels(text: str) -> list[str]:
    """Return each individual ``S#`` label referenced in ``text``, in
    order of first appearance, de-duplicated. Grouped citations like
    "[S1, S3]" expand to individual labels."""
    seen: dict[str, None] = {}
    for group in _LABEL_GROUP_RE.findall(text):
        for label in _LABEL_RE.findall(group):
            seen.setdefault(label, None)
    return list(seen)


def resolve_citations(text: str, packet_passages: list[dict]) -> list[Citation]:
    """Return one Citation per unique label referenced in ``text``."""
    by_label = {p["label"]: p for p in packet_passages}
    citations: list[Citation] = []
    for label in extract_labels(text):
        passage = by_label.get(label)
        if passage is None:
            citations.append(Citation(label=label, unresolved=True))
            continue
        span = None
        if "start" in passage and "end" in passage:
            span = (passage["start"], passage["end"])
        citations.append(
            Citation(
                label=label,
                passage_id=passage.get("id"),
                title=passage.get("title"),
                path=passage.get("path"),
                span=span,
                unresolved=False,
            )
        )
    return citations


def render_evidence(packet: dict) -> str:
    """Render a retrieval packet's passages as an evidence block, one
    line/block per passage, marking Q&A-kind passages with "[Q&A]"."""
    lines = []
    for passage in packet.get("passages", []):
        marker = " [Q&A]" if passage.get("kind") == "qa" else ""
        lines.append(f"[{passage['label']}]{marker} {passage.get('text', '')}")
    return "\n".join(lines)

"""Host-driven "Did you mean X?" parsing/classification for unknown,
likely-misspelled words (owner request 2026-09-21; see
tutor.app.agent_loop's clarify wiring and HANDOFF.md).

Students are children who spell phonetically ("ardweeno" for Arduino).
Retrieval's own spelling fallback (edit distance <= 2) cannot reach these,
so the host runs a small constrained LLM call asking what the student most
likely meant, then gates the answer against the library before ever
showing it. Everything in this module is pure text handling -- no I/O, no
retrieval, no LLM calls -- so it can be unit tested directly.
"""

from __future__ import annotations

import re

_NAME_STRIP_CHARS = "\"'*"

_YES_REPLIES = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "ya",
    "y",
    "yes please",
    "thats it",
    "correct",
    "right",
}

_NO_REPLIES = {
    "no",
    "nope",
    "nah",
    "n",
    "not that",
    "no thats not it",
}

_DESCRIPTION_MAX_WORDS = 12
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')


def _clean_name(raw: str) -> str:
    name = raw.strip()
    name = name.strip(_NAME_STRIP_CHARS + " ")
    name = name.rstrip(".").strip()
    return name


# Function words a clipped description must not end on ("...used for
# building" reads as cut off; "...used for building electronics" is fine).
_DANGLING_TAIL_WORDS = frozenset(
    {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "by",
     "used", "that", "which", "from", "as", "at", "is", "are", "its"}
)


def _clip_words(text: str, max_words: int) -> str:
    """Clip to ``max_words``, drop any dangling function word the clip
    left at the end, strip a trailing period, and lower-case the first
    letter so it reads inline after "Did you mean Arduino, ...".
    """
    words = text.split()[:max_words]
    while words and words[-1].lower().rstrip(".,;:") in _DANGLING_TAIL_WORDS:
        words.pop()
    out = " ".join(words).rstrip(".,;:")
    if out and not (len(out) > 1 and out[1].isupper()):  # keep acronyms like "LED"
        out = out[0].lower() + out[1:]
    return out


def parse_candidate(raw: str) -> tuple[str, str] | None:
    """Parse the clarify LLM call's raw one-line reply into ``(name,
    description)``, or ``None`` if nothing usable is there.

    Preferred shape: ``NAME | five-word description`` -- split on the
    first ``|``, the name stripped of surrounding quotes/asterisks/a
    trailing period and required to be 1-4 words, the description clipped
    to at most 12 words. With no ``|``: a quoted name (``The student likely
    meant "dinosaur"``) is accepted with an empty description, and
    otherwise a short (<=4 word) line is accepted as the name outright.
    Anything longer or unparseable returns ``None``.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()

    if "|" in text:
        name_part, _, desc_part = text.partition("|")
        name = _clean_name(name_part)
        name_words = name.split()
        if not name or not (1 <= len(name_words) <= 4):
            return None
        description = _clip_words(desc_part.strip(), _DESCRIPTION_MAX_WORDS)
        return name, description

    match = _QUOTED_RE.search(text)
    if match:
        candidate = match.group(1) or match.group(2)
        name = _clean_name(candidate)
        name_words = name.split()
        if name and 1 <= len(name_words) <= 4:
            return name, ""
        return None

    words = text.split()
    if 1 <= len(words) <= 4:
        name = _clean_name(text)
        if name:
            return name, ""
    return None


def classify_reply(text: str | None) -> str | None:
    """Classify a student's reply to the host's own "Did you mean X?"
    yes/no question as ``"yes"``, ``"no"``, or ``None`` (unmatched -- a
    description, a re-typed word, or anything else).

    CLAUDE.md forbids word-list heuristics for *detecting follow-ups* in
    general (a student's grammar/spelling can't be trusted to signal a
    reference). This is different: the host itself just asked a plain
    yes/no question, so matching the reply against a small, host-editable
    yes/no vocabulary is a legitimate, narrow classifier for THAT
    question -- and anything that doesn't match falls through safely to
    being treated as a description/new input rather than being
    misclassified.
    """
    if not text:
        return None
    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in _YES_REPLIES:
        return "yes"
    if cleaned in _NO_REPLIES:
        return "no"
    return None

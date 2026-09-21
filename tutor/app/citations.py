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

# 2026-09-21 dump-collapse bug fix: label-stacking only counts as a dump
# signal when the whole answer is long enough to plausibly be a "here's
# everything" evasion. A short, correct answer like "Yes, DNA is a
# molecule. [S1][S2][S3][S4]" carries several stacked labels but is not a
# dump -- a genuine dump is long. Word count of the whole answer text.
_MIN_DUMP_ANSWER_WORDS = 60

# Reserved label for the synthetic seed exchange (tutor.app.seed_exchange):
# real evidence numbering always starts at S1 (Session.allocate_label), so
# "S0" never names a real passage. The resolver refuses it unconditionally
# (never resolves, never opens the source viewer) and attribute_sentences
# never attributes a sentence to it, even if a caller's passage list
# somehow contains an id labelled "S0".
RESERVED_SEED_LABEL = "S0"


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


# 2026-09-21 specifics gate (owner-reported LIVE bug): the tutor fabricated
# a person ("Vitus Andronicus") and two temperatures, and the host marked
# both sentences as "found in the sources" purely on loose word overlap
# with a real cold/temperature passage ("cold", "survived", "temperatures"
# etc). Loose overlap alone is too weak once a sentence carries its own
# SPECIFIC claims -- a number, or a proper name -- since those are exactly
# what a fabrication invents while still sharing plenty of generic
# vocabulary with a real passage. Fix: such a sentence may only be marked
# supported when every specific it carries is actually present in the
# evidence text (see ``_specifics_supported``); a sentence with no
# specifics at all keeps the plain overlap rule from before.
_NUM_THEN_PAREN_RE = re.compile(r"[-−]?\d[\d,]*(?:\.\d+)?[^\s(),]*\s*\(([^()]*)\)")

_MULTIWORD_NAME_RE = re.compile(r"\b[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)+\b")
_SINGLE_CAP_WORD_RE = re.compile(r"\b[A-Z][A-Za-z'-]*\b")

# Small, intentionally short stoplist (spec: "fine to skip [nationalities]
# if hard") -- months, days, and the pronoun "I" are cheap and common
# enough to be worth excluding explicitly.
_NAME_STOPLIST = frozenset(
    {
        "I",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    }
)


def _exempt_conversion_numbers(sentence: str) -> set[str]:
    """Figures that appear inside parentheses immediately following another
    number (+ optional unit letters), e.g. the "-94" in "-70°C
    (-94°F✓ checked)" -- a unit-conversion OUTPUT riding along
    with its source number does not need its own source; the ORIGINAL
    number (the "-70" here) still does."""
    exempt: set[str] = set()
    for m in _NUM_THEN_PAREN_RE.finditer(sentence):
        exempt |= set(_FIGURE_RE.findall(_normalize_minus(m.group(1))))
    return exempt


def _specific_numbers(sentence: str) -> set[str]:
    return _figures(sentence) - _exempt_conversion_numbers(sentence)


def _specific_names(sentence: str) -> list[str]:
    """Capitalised multi-word names, and single capitalised words that are
    not sentence-initial and not in ``_NAME_STOPLIST``."""
    names: list[str] = []
    covered: list[tuple[int, int]] = []
    for m in _MULTIWORD_NAME_RE.finditer(sentence):
        names.append(m.group(0))
        covered.append(m.span())
    first_word = next(re.finditer(r"\S+", sentence), None)
    first_start = first_word.start() if first_word else -1
    for m in _SINGLE_CAP_WORD_RE.finditer(sentence):
        if m.start() == first_start:
            continue
        if any(m.start() >= s and m.end() <= e for s, e in covered):
            continue
        word = m.group(0)
        if word in _NAME_STOPLIST:
            continue
        names.append(word)
    return names


def _name_found_in_evidence(name: str, evidence_text: str) -> bool:
    """A multi-word name matches whole, case-insensitively; failing that, a
    surname-only/one-token match counts as long as that token appears as a
    whole word (spec: "allow a surname-only or one-token match")."""
    if re.search(rf"\b{re.escape(name)}\b", evidence_text, re.IGNORECASE):
        return True
    for token in name.split():
        if re.search(rf"\b{re.escape(token)}\b", evidence_text, re.IGNORECASE):
            return True
    return False


def _specifics_supported(sentence: str, evidence_text: str) -> bool:
    """True if every specific (number, proper name) that ``sentence``
    states is present in ``evidence_text``. A sentence with no specifics at
    all trivially passes -- see ``is_supported``."""
    evidence_figures = _figures(evidence_text)
    for number in _specific_numbers(sentence):
        if number not in evidence_figures:
            return False
    for name in _specific_names(sentence):
        if not _name_found_in_evidence(name, evidence_text):
            return False
    return True


def is_supported(sentence: str, passage_text: str, all_passages_text: str = "") -> bool:
    """Mechanical, deterministic support check: does ``sentence`` share
    enough content terms with ``passage_text`` to plausibly be drawn from
    it, AND (2026-09-21 specifics gate) does every SPECIFIC the sentence
    states -- a number or a proper name -- actually appear somewhere in
    this turn's evidence (``passage_text`` plus ``all_passages_text``, the
    concatenation of every passage in the current packet -- never earlier
    TUTOR text, which is not evidence)?

    Overlap rule (documented, not tuned on live data): supported if the
    sentence and passage share at least ``_MIN_SHARED_TERMS`` content terms
    (numbers count as terms -- ``tokenize`` keeps digit runs) OR the shared
    terms are at least ``_MIN_SHARED_FRACTION`` of the sentence's own
    content terms. A sentence that merely quotes the passage back still
    counts as supported by this rule (see ``detect_evidence_dump`` for
    catching wholesale copying).

    Specifics gate: a sentence with no numbers and no proper names keeps
    exactly today's overlap-only behavior. A sentence that DOES state a
    number or a name is only supported when every one of those specifics
    is found in the evidence text -- this is what stops a fabricated name
    ("Vitus Andronicus") or a fabricated number riding on real-sounding
    generic vocabulary ("cold", "survived", "temperatures") from being
    marked as found in the sources."""
    sentence_terms = set(tokenize(sentence))
    if not sentence_terms:
        return False
    passage_terms = set(tokenize(passage_text))
    shared = sentence_terms & passage_terms
    if len(shared) >= _MIN_SHARED_TERMS:
        overlap_ok = True
    else:
        overlap_ok = (len(shared) / len(sentence_terms)) >= _MIN_SHARED_FRACTION
    if not overlap_ok:
        return False
    evidence_text = passage_text + "\n" + all_passages_text
    return _specifics_supported(sentence, evidence_text)


def _dump_signal(text: str, packet_passages: list[dict]) -> tuple[bool, set[str]]:
    """Shared evidence-dump detection for ``detect_evidence_dump`` and
    ``resolve_citations``'s support override below. Two mechanical,
    deterministic signals (2026-09-20 evidence-dump follow-up, see
    docs/citation_experiment.md):

    1. Label-stacking: any single sentence/bullet carries
       ``_MIN_STACKED_LABELS`` (4) or more distinct ``[S#]`` labels -- the
       hallmark of a "here's everything" trailer sentence like
       "...[S1][S2]...[S11]". Every label on such a sentence is flagged.
       Only counted when the whole answer is at least
       ``_MIN_DUMP_ANSWER_WORDS`` words long -- a short, correct answer can
       legitimately cite several sources on its one sentence without being
       a dump (2026-09-21 fix: "Yes, DNA is a molecule. [S1][S2][S3][S4]"
       was wrongly flagged and its whole bubble hidden by the UI).
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
    is_long_answer = len(text.split()) >= _MIN_DUMP_ANSWER_WORDS
    if is_long_answer:
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
    """Return one Citation per unique label referenced in ``text``.

    ``RESERVED_SEED_LABEL`` ("S0") always resolves as unresolved, even if
    ``packet_passages`` contains an entry labelled "S0" -- it is reserved
    for the synthetic seed exchange and must never be citable for a real
    question."""
    by_label = {
        p["label"]: p for p in packet_passages if p.get("label") != RESERVED_SEED_LABEL
    }
    # Evidence set for the specifics gate (is_supported): every passage in
    # this turn's packet, not just the one label happens to resolve to --
    # a specific backed by a sibling passage still counts.
    all_passages_text = "\n".join(p.get("text", "") for p in packet_passages)
    _, flagged_labels = _dump_signal(text, packet_passages)
    citations: list[Citation] = []
    for label in extract_labels(text):
        if label == RESERVED_SEED_LABEL:
            citations.append(Citation(label=label, unresolved=True, supported=False))
            continue
        passage = by_label.get(label)
        if passage is None:
            citations.append(Citation(label=label, unresolved=True, supported=False))
            continue
        span = None
        if "start" in passage and "end" in passage:
            span = (passage["start"], passage["end"])
        sentence = _sentence_for_label(text, label)
        supported = is_supported(sentence, passage.get("text", ""), all_passages_text)
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

# Used instead of ``_CITATION_REMINDER`` when ``app.model_writes_citations``
# is False (default -- see docs/attribution_design.md, "Model-written
# labels are no longer requested"): the host's per-sentence attribution
# already links each sentence to the passage that backs it via
# ``attribute_sentences``/host-inferred citation chips, independent of
# whether the model itself writes an ``[S#]`` label, so asking the model
# to write labels is no longer necessary. Keeps the two other reminders
# ("do not list/copy the sources", "say so if none of them answer") that
# have nothing to do with the model writing a label itself.
_NO_LABEL_REMINDER = (
    "Do not list or copy the sources. If none of them answers the "
    "question, say so."
)


def citation_reminder_text(model_writes_citations: bool) -> str:
    """Return the evidence-tail reminder line for the given
    ``app.model_writes_citations`` setting -- ``True`` reproduces the old
    (``_CITATION_REMINDER``) bytes exactly."""
    return _CITATION_REMINDER if model_writes_citations else _NO_LABEL_REMINDER


_FIGURE_RE = re.compile(r"[-−]?\d+(?:\.\d+)?")

_NON_CLAIM_MAX_WORDS = 4


def _normalize_minus(text: str) -> str:
    """ASCII hyphen-minus and Unicode minus (U+2212) are the same figure."""
    return text.replace("−", "-")


_LEADING_LIST_NUMBER_RE = re.compile(r"^\s*\d+[.)](?:\s+|$)")

# One or more bracketed tokens and nothing else -- e.g. "[Q&A]" or "[S1]"
# left standing alone after sentence splitting.
_BRACKET_ONLY_RE = re.compile(r"^(?:\[[^\[\]]*\]\s*)+$")

# Invented, non-[S#] "citation-shaped" tokens the model sometimes tacks on
# that are NOT real evidence labels (2026-09-20 owner-reported live bug:
# a trailing "[Cite: [Q&A]]" line even got its own spurious "not found"
# marker in the UI, and "... found in interstellar space. [Q&A]" showed
# the bogus label as literal text). This is the single source of truth for
# the pattern; tutor/ui/app.js mirrors it (see the comment there) since the
# UI must not display these either. Kept as a short, explicit, case-
# insensitive list -- never a broad "any bracket" match -- so ordinary
# bracketed text ("[H2O]", "[1, 2, 3]") is never touched. "[Cite: ...]"
# allows one level of nesting so "[Cite: [Q&A]]" matches as a whole.
_INVENTED_LABEL_RE = re.compile(
    r"\[\s*Q\s*&\s*A\s*\]"
    r"|\[\s*Cite\s*:\s*(?:\[[^\[\]]*\]|[^\[\]]*)\]"
    r"|\[\s*Source\s*\]"
    r"|\[\s*Sources\s*:[^\[\]]*\]"
    r"|\[\s*citation\s+needed\s*\]",
    re.IGNORECASE,
)


# 2026-09-21 markdown-link fake citation (owner-reported LIVE bug, after
# 8f26f82 + f0f8755 hid the plain "[Cite: [Q&A]]" / fake "Sources:" forms):
# the model now ends answers with a fake citation in MARKDOWN-LINK shape
# pointing at the live internet -- this is an OFFLINE tool, the URL is
# invented -- e.g. "[Cite: [S1]](https://en.wikipedia.org/wiki/Titin)".
# `_INVENTED_LABEL_RE` alone matches only the "[Cite: [S1]]" part and
# leaves "(https://...)" behind as apparent claim text. `_BRACKET_TOKEN_RE`
# matches ANY top-level bracketed token (one level of nesting allowed, same
# as "[Cite: [S1]]" above) so this catches the general shape -- "[Cite:
# ...](url)", "[Source](url)", "[S1](url)" -- without a broad "any
# bracket" rule: a URL suffix is required, so ordinary bracketed text like
# "[H2O]", "[1, 2, 3]", a trailing "[S1]" with no URL, or "f(x)"/"(see
# above)" parentheses after ordinary brackets are never touched. This is
# the single source of truth for the pattern; tutor/ui/app.js mirrors it
# (see `INVENTED_LINK_RE` there) -- keep the two in sync.
_BRACKET_TOKEN_RE = r"\[(?:[^\[\]]|\[[^\[\]]*\])*\]"
_INVENTED_LINK_RE = re.compile(_BRACKET_TOKEN_RE + r"\s*\(\s*https?://[^\s()]*\s*\)")


def collapse_invented_links(text: str) -> str:
    """Collapses every "invented-label-shaped bracket token immediately
    followed by a parenthesised http(s) URL" (see ``_INVENTED_LINK_RE``) in
    ``text`` into just the real ``[S#]`` label(s) it wraps, if any -- e.g.
    "[Cite: [S1]](https://...)" -> "[S1]" -- or into "" if it wraps no real
    label at all -- e.g. "[Source](https://...)" -> "". Display/analysis
    helper only: never mutates the stored answer text itself. Mirrored in
    tutor/ui/app.js as ``collapseInventedLinks``."""

    def _replace(match: re.Match[str]) -> str:
        labels = list(dict.fromkeys(_LABEL_RE.findall(match.group(0))))
        return "[" + ", ".join(labels) + "]" if labels else ""

    return _INVENTED_LINK_RE.sub(_replace, text)


# 2026-09-21 model-authored source-list block (owner-reported live bug): the
# model sometimes writes its OWN fake "Sources:" list with invented passage
# descriptions instead of relying on the real [S#] chips the app already
# renders. This is the single source of truth for detecting such a block;
# tutor/ui/app.js mirrors it (see the comment there) so the UI never renders
# the heading or the label-led lines that follow it either. Kept
# conservative: a heading line ("Sources:" / "Source:" / "References:" /
# "Citations:", optional markdown emphasis/heading marks) must be
# IMMEDIATELY followed by one or more consecutive lines that each START with
# a "[S#]" citation label -- a normal sentence that merely starts with or
# contains "[S1]" never matches this, and a bare heading with no label-led
# lines after it is left alone.
_SOURCE_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?\*{0,2}\s*(?:Sources?|References?|Citations?)\s*:?\s*\*{0,2}\s*$",
    re.IGNORECASE,
)

_LABEL_LED_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]")


def _source_block_line_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Line-index ranges ``[start, end)`` (end exclusive) of each detected
    model-authored source-list block: a heading line plus every consecutive
    label-led line that immediately follows it. A heading with no label-led
    line right after it produces no range at all."""
    ranges: list[tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        if _SOURCE_HEADING_RE.match(lines[i]):
            j = i + 1
            while j < n and _LABEL_LED_LINE_RE.match(lines[j]):
                j += 1
            if j > i + 1:
                ranges.append((i, j))
                i = j
                continue
        i += 1
    return ranges


def find_model_source_block_ranges(text: str) -> list[tuple[int, int]]:
    """Character-offset ``(start, end)`` spans into ``text`` of every
    model-authored source-list block detected (see ``_source_block_line_ranges``).
    Used both to exclude the block from attribution sentence units
    (``_sentence_spans``) and, mirrored in tutor/ui/app.js, to hide it from
    the rendered chat bubble -- the host never edits the stored answer text
    itself, only what a display layer builds from it."""
    lines = text.split("\n")
    line_starts: list[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1

    spans: list[tuple[int, int]] = []
    for start_idx, end_idx in _source_block_line_ranges(lines):
        block_start = line_starts[start_idx]
        last_line_idx = end_idx - 1
        block_end = line_starts[last_line_idx] + len(lines[last_line_idx])
        spans.append((block_start, block_end))
    return spans


def _strip_label_noise(text: str) -> str:
    """Strip ``[S123]``-style citation labels and a leading list-numbering
    marker ("1. ", "2) ") before scanning for figures -- otherwise the
    label's own digits (or a list item's ordinal) get flagged as an
    "unbacked number" the model never actually claimed. Measured:
    docs/soak_v3_analysis.md §6 -- most `unbacked_number` flags in the
    turns-31-33 loops were the sentence's own trailing ``[S12]``/``[S18]``
    label digits, not content."""
    text = _LABEL_GROUP_RE.sub("", text)
    text = _LEADING_LIST_NUMBER_RE.sub("", text)
    return text


def _figures(text: str) -> set[str]:
    """The set of number tokens (sign + digits + optional decimal) in
    ``text``, with Unicode minus normalized to ASCII hyphen-minus, and
    ``[S#]`` label digits / a leading list-number marker stripped first
    (see ``_strip_label_noise``)."""
    return set(_FIGURE_RE.findall(_normalize_minus(_strip_label_noise(text))))


# Markdown table row / separator (2026-09-20 structure-aware attribution
# follow-up, tables feedback): a pipe-delimited row, and the "|---|---|"
# style separator row that follows a header. A list item line ("- foo" or
# "1. foo").
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*\|?\s*$")
_LIST_ITEM_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")


def _trimmed_span(text: str, s: int, e: int) -> tuple[int, int] | None:
    chunk = text[s:e]
    lstrip = len(chunk) - len(chunk.lstrip())
    rstrip = len(chunk) - len(chunk.rstrip())
    ns, ne = s + lstrip, e - rstrip
    return (ns, ne) if ns < ne else None


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence/bullet-line/table-row spans into the ORIGINAL ``text``.

    A markdown table ROW and a list item are each treated as ONE unit --
    never split at "1." or inside a cell -- so no span boundary can fall
    inside list numbering or land on pipe/``<br>`` table syntax alone. A
    table's header row and its ``---`` separator row are non-claims and
    produce no span at all. Everything else (plain paragraph text) is
    split the same way ``_sentences`` always has been, on sentence
    terminators or blank lines, keeping character offsets (leading/
    trailing whitespace trimmed from each span)."""
    lines = text.split("\n")
    line_starts: list[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1

    # 2026-09-21 model-authored source-list block: a heading line + the
    # label-led lines under it never become sentence/attribution units at
    # all (see find_model_source_block_ranges).
    skip_line_ranges = _source_block_line_ranges(lines)

    def _in_skip_range(idx: int) -> tuple[int, int] | None:
        for r_start, r_end in skip_line_ranges:
            if r_start <= idx < r_end:
                return (r_start, r_end)
        return None

    spans: list[tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        skip_range = _in_skip_range(i)
        if skip_range is not None:
            i = skip_range[1]
            continue
        if _TABLE_ROW_RE.match(line) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            # Header row + separator row: both non-claims, no span at all.
            i += 2
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                row_span = _trimmed_span(text, line_starts[i], line_starts[i] + len(lines[i]))
                if row_span is not None:
                    spans.append(row_span)
                i += 1
            continue
        if _LIST_ITEM_LINE_RE.match(line):
            item_span = _trimmed_span(text, line_starts[i], line_starts[i] + len(line))
            if item_span is not None:
                spans.append(item_span)
            i += 1
            continue

        # Paragraph: gather consecutive non-table/non-list lines and split
        # them the old punctuation-based way, preserving offsets.
        j = i
        while j < n:
            nxt = lines[j]
            if _in_skip_range(j) is not None:
                break
            if _LIST_ITEM_LINE_RE.match(nxt):
                break
            if _TABLE_ROW_RE.match(nxt) and j + 1 < n and _TABLE_SEP_RE.match(lines[j + 1]):
                break
            j += 1
        para_start = line_starts[i]
        para_end = line_starts[j - 1] + len(lines[j - 1]) if j > i else para_start
        para_text = text[para_start:para_end]
        bounds = []
        start = 0
        for m in _SENTENCE_SPLIT_RE.finditer(para_text):
            bounds.append((start, m.start()))
            start = m.end()
        bounds.append((start, len(para_text)))
        for s, e in bounds:
            span = _trimmed_span(para_text, s, e)
            if span is not None:
                spans.append((para_start + span[0], para_start + span[1]))
        i = j if j > i else i + 1
    return spans


def _is_short_non_claim(sentence: str) -> bool:
    """A question to the student, or a short exclamation like "Great
    question!" -- neither a claim to attribute nor one to flag unbacked."""
    s = sentence.strip()
    if not s:
        return True
    if s.endswith("?"):
        return True
    if s.endswith("!") and len(re.findall(r"\w+", s)) <= _NON_CLAIM_MAX_WORDS:
        return True
    # A bare list-numbering marker ("1.", "2)") split off as its own
    # "sentence" by the punctuation-based splitter -- not a claim itself.
    if _LEADING_LIST_NUMBER_RE.match(s) and _LEADING_LIST_NUMBER_RE.sub("", s) == "":
        return True
    # A fragment that is nothing but one or more bracketed tokens, AND
    # none of them is a real [S#] citation label, carries no claim of its
    # own -- e.g. a bogus trailing "[Q&A]" the model tacked on (2026-09-20
    # bracket-label follow-up: this used to get its own spurious unbacked
    # ("tutor's own words") marker in the UI). A bracket-only fragment that
    # DOES carry a real [S#] label (e.g. a lone trailing "[S1]") is left
    # alone -- that is a legitimate citation, handled by `_labels_in`
    # below, not noise.
    if _BRACKET_ONLY_RE.match(s) and not _LABEL_RE.search(s):
        return True
    # A fragment made of nothing but invented, non-[S#] "citation-shaped"
    # tokens (see _INVENTED_LABEL_RE) once those are stripped away -- e.g.
    # a bogus trailing "[Cite: [Q&A]]" line -- carries no claim either,
    # even when it does not parse as _BRACKET_ONLY_RE (nested brackets).
    if _INVENTED_LABEL_RE.sub("", s).strip() == "":
        return True
    # A fragment that is nothing but a "[Cite: ...](url)"/"[Source](url)"
    # style invented-link token (see collapse_invented_links) wrapping NO
    # real [S#] label -- e.g. a bogus trailing
    # "[Source](https://en.wikipedia.org/wiki/Rubber)" line -- carries no
    # claim either. One that DOES wrap a real label (e.g. "[Cite:
    # [S1]](url)") collapses to just "[S1]" here, which is left alone --
    # `_labels_in`/the resolvable-citation path below handles it as a real
    # citation, same as a bare trailing "[S1]".
    if collapse_invented_links(s).strip() == "":
        return True
    return False


def _labels_in(sentence: str) -> list[str]:
    labels: dict[str, None] = {}
    for group in _LABEL_GROUP_RE.findall(sentence):
        for label in _LABEL_RE.findall(group):
            labels.setdefault(label, None)
    return list(labels)


def _best_supporting_passage(sentence: str, passages: list[dict]) -> tuple[dict | None, int]:
    """The passage (if any) that best supports ``sentence`` by ``is_supported``,
    broken by most shared content terms (reuses the tokenizer/overlap rule
    ``is_supported`` already uses -- not duplicated here)."""
    sentence_terms = set(tokenize(sentence))
    all_passages_text = "\n".join(p.get("text", "") for p in passages)
    best: dict | None = None
    best_score = -1
    for passage in passages:
        text = passage.get("text", "")
        if not is_supported(sentence, text, all_passages_text):
            continue
        shared = sentence_terms & set(tokenize(text))
        score = len(shared)
        if score > best_score:
            best_score = score
            best = passage
    return best, best_score


@dataclass(frozen=True)
class Attribution:
    sentence_span: tuple[int, int]
    passage_id: str | None
    label: str
    score: float
    model_cited: bool


@dataclass(frozen=True)
class UnbackedSpan:
    span: tuple[int, int]
    reason: str


@dataclass(frozen=True)
class AttributionResult:
    attributions: list[Attribution]
    unbacked_spans: list[UnbackedSpan]


def attribute_sentences(answer: str, passages: list[dict]) -> AttributionResult:
    """Attribute each sentence/bullet-line of ``answer`` to the passage (if
    any) it is drawn from, without ever modifying ``answer`` or fabricating
    a ``[S#]`` label it does not itself carry (per §11's "host never
    fabricates a citation" rule -- see docs/attribution_design.md).

    A sentence that itself carries a resolvable ``[S#]`` label is attributed
    to that passage with ``model_cited=True``. An unlabeled sentence is
    attributed (``model_cited=False``) to whichever given passage best
    supports it by the same overlap rule ``is_supported``/``resolve_citations``
    use, if any does. Everything else lands in ``unbacked_spans``, flagged
    ``"unbacked_number"`` instead of plain ``"unbacked"`` when the sentence
    contains a figure that appears in none of the (non-empty) passages
    supplied -- an invented number is riskier than a generic own-example. A
    short non-claim (a question to the student, a short exclamation) lands
    in neither list. Pure function: ``answer`` is never modified and spans
    index the original string exactly.
    """
    by_label = {p["label"]: p for p in passages if p.get("label") != RESERVED_SEED_LABEL}
    all_passages_text = "\n".join(p.get("text", "") for p in passages)
    attributions: list[Attribution] = []
    unbacked: list[UnbackedSpan] = []

    for start, end in _sentence_spans(answer):
        sentence = answer[start:end]
        if _is_short_non_claim(sentence):
            continue

        # 2026-09-21 real-list-item follow-up (owner-reported live bug B): a
        # label RESOLVING to a passage in the packet is not the same as that
        # passage actually SUPPORTING the claim -- e.g. a numbered-list item
        # citing [S1] when S1's passage shares no content with the claim. A
        # resolvable label only short-circuits to model_cited=True when at
        # least one of its passages actually supports the sentence (same
        # `is_supported` overlap rule `resolve_citations` uses); otherwise
        # the sentence falls through to the same best-match/unbacked logic
        # as an unlabeled sentence, so the UI can still show its ○ "not
        # found" marker next to the model's own (unresolved-looking) chip.
        resolvable = [(lbl, by_label[lbl]) for lbl in _labels_in(sentence) if lbl in by_label]
        # A "sentence" unit that is nothing but the label itself (e.g. a
        # trailing "[S1]" split off as its own fragment by the punctuation
        # splitter) carries no claim text of its own to check for support --
        # trust it as a legitimate citation like before. A unit that DOES
        # carry its own claim text (a real sentence or list item with an
        # inline [S#]) only counts as model_cited when at least one of its
        # labels' passages actually supports that claim.
        # 2026-09-21 markdown-link fake citation follow-up: collapse any
        # "[Cite: [S1]](url)"/"[Source](url)" invented-link wrapper down to
        # its bare real label(s) (or "") FIRST, so the URL/wrapper text
        # (e.g. "en wikipedia org wiki Titin") is never mistaken for the
        # sentence's own claim text -- otherwise a line that is really just
        # a citation gets treated as an unsupported claim and flagged
        # unbacked (see collapse_invented_links).
        has_own_claim_text = bool(
            tokenize(_LABEL_GROUP_RE.sub("", collapse_invented_links(sentence)))
        )
        if not has_own_claim_text:
            supported_resolvable = resolvable
        else:
            supported_resolvable = [
                (lbl, passage)
                for lbl, passage in resolvable
                if is_supported(sentence, passage.get("text", ""), all_passages_text)
            ]
        if supported_resolvable:
            for _label, passage in supported_resolvable:
                attributions.append(
                    Attribution(
                        sentence_span=(start, end),
                        passage_id=passage.get("id"),
                        label=passage["label"],
                        score=1.0,
                        model_cited=True,
                    )
                )
            continue

        best, score = _best_supporting_passage(sentence, passages)
        if best is not None:
            attributions.append(
                Attribution(
                    sentence_span=(start, end),
                    passage_id=best.get("id"),
                    label=best["label"],
                    score=float(score),
                    model_cited=False,
                )
            )
            continue

        reason = "unbacked"
        if passages:
            sentence_figures = _figures(sentence)
            if sentence_figures:
                passage_figures: set[str] = set()
                for passage in passages:
                    passage_figures |= _figures(passage.get("text", ""))
                if not (sentence_figures & passage_figures):
                    reason = "unbacked_number"
        unbacked.append(UnbackedSpan(span=(start, end), reason=reason))

    return AttributionResult(attributions=attributions, unbacked_spans=unbacked)


def render_evidence(packet: dict, *, model_writes_citations: bool = True) -> str:
    """Render a retrieval packet's passages as an evidence block, one
    line/block per passage (label at the start), marking Q&A-kind
    passages with "[Q&A]", followed by a short citation reminder line --
    unless the packet has no passages, in which case an empty string is
    returned (no reminder to cite evidence that was never shown).

    ``model_writes_citations`` (default True -- see ``app.
    model_writes_citations``, docs/attribution_design.md) selects between
    the full reminder (which asks the model to write an ``[S#]`` label)
    and the shorter one that only says not to list/copy the sources and
    to say so if none answer the question. Passages are still labelled
    ``[S1]``... in the evidence block either way -- the host and UI need
    the ids regardless of whether the model is asked to write them."""
    passages = packet.get("passages", [])
    if not passages:
        return ""
    lines = []
    for passage in passages:
        marker = " [Q&A]" if passage.get("kind") == "qa" else ""
        lines.append(f"[{passage['label']}]{marker} {passage.get('text', '')}")
    if packet.get("note"):
        lines.append(packet["note"])
    lines.append(citation_reminder_text(model_writes_citations))
    return "\n".join(lines)

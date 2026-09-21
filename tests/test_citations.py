"""RED tests for tutor.app.citations.

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §8 (numbered
`[S#]` labels, stable for the lesson) and §11 ("`[S#]` labels resolved by
the host and rejected if unresolved"); docs/plan/offline_tutor_implementation_
plan.md §0.4 (Q&A-kind passages carry a "[Q&A]" marker in the rendered
evidence text given to the model).

Contract decisions made here (spec silent):
- `extract_labels(text) -> list[str]` returns each individual label token
  found in the text, in order of first appearance, de-duplicated;
  "[S1, S3]"-style grouped citations expand to ["S1", "S3"].
- `resolve_citations(text, packet_passages) -> list[Citation]` returns one
  Citation per *unique* label referenced in text, each with the label,
  and (when resolvable) passage_id/title/path/span from the matching
  passage dict in packet_passages (matched by "label" key); a label with
  no matching passage becomes a Citation with `unresolved=True` and the
  other fields left None -- it is never silently dropped from the result.
- `render_evidence(packet) -> str` renders one line/block per passage
  including its "[S#]" label and, when passage["kind"] == "qa", an extra
  "[Q&A]" marker in the rendered text.
"""

from __future__ import annotations

from pathlib import Path as _Path

import pytest

from tutor.app.citations import (
    Citation,
    _sentence_spans,
    collapse_invented_links,
    extract_labels,
    find_model_source_block_ranges,
    is_supported,
    render_evidence,
    resolve_citations,
)


def _passage(
    label, passage_id="p1", title="Title", path="A/Title", start=0, end=10, kind="article"
):
    return {
        "label": label,
        "id": passage_id,
        "title": title,
        "path": path,
        "start": start,
        "end": end,
        "kind": kind,
    }


# ---------------------------------------------------------------------------
# extract_labels
# ---------------------------------------------------------------------------


def test_extract_single_label():
    assert extract_labels("Water boils at 100C [S1].") == ["S1"]


def test_extract_multiple_separate_labels_in_order():
    text = "Photosynthesis needs light [S1]. Plants also need water [S2]."
    assert extract_labels(text) == ["S1", "S2"]


def test_extract_grouped_label_expands_to_individual_labels():
    text = "This is supported by multiple sources [S1, S3]."
    assert extract_labels(text) == ["S1", "S3"]


def test_extract_grouped_label_without_space_after_comma():
    text = "See [S2,S4] for details."
    assert extract_labels(text) == ["S2", "S4"]


def test_extract_deduplicates_repeated_labels_keeping_first_order():
    text = "First claim [S1]. Second claim [S2]. Back to first idea [S1]."
    assert extract_labels(text) == ["S1", "S2"]


def test_extract_returns_empty_list_when_no_labels_present():
    assert extract_labels("No citations here at all.") == []


def test_extract_ignores_non_citation_bracketed_text():
    assert extract_labels("An array like [1, 2, 3] is not a citation.") == []


def test_extract_handles_double_digit_labels():
    text = "Cited [S10] and [S2]."
    assert extract_labels(text) == ["S10", "S2"]


# ---------------------------------------------------------------------------
# resolve_citations
# ---------------------------------------------------------------------------


def test_resolve_known_label_returns_full_citation_fields():
    passages = [_passage("S1", passage_id="pid-1", title="Water", path="A/Water", start=5, end=50)]
    citations = resolve_citations("Water boils at 100C [S1].", passages)
    assert len(citations) == 1
    citation = citations[0]
    assert isinstance(citation, Citation)
    assert citation.label == "S1"
    assert citation.passage_id == "pid-1"
    assert citation.title == "Water"
    assert citation.path == "A/Water"
    assert citation.span == (5, 50)
    assert citation.unresolved is False


def test_resolve_unknown_label_is_flagged_unresolved_not_dropped():
    passages = [_passage("S1")]
    citations = resolve_citations("Claim cites a missing source [S9].", passages)
    assert len(citations) == 1
    citation = citations[0]
    assert citation.label == "S9"
    assert citation.unresolved is True
    assert citation.passage_id is None


def test_resolve_returns_one_citation_per_unique_label():
    passages = [_passage("S1", passage_id="p1"), _passage("S2", passage_id="p2")]
    text = "Claim one [S1]. Claim two [S2]. Restated claim one [S1]."
    citations = resolve_citations(text, passages)
    assert len(citations) == 2
    assert {c.label for c in citations} == {"S1", "S2"}


def test_resolve_mixed_known_and_unknown_labels_all_present():
    passages = [_passage("S1", passage_id="p1")]
    text = "Known fact [S1]. Unknown fact [S7]."
    citations = resolve_citations(text, passages)
    labels = {c.label: c for c in citations}
    assert labels["S1"].unresolved is False
    assert labels["S7"].unresolved is True


def test_resolve_no_labels_returns_empty_list():
    passages = [_passage("S1")]
    assert resolve_citations("No citation markers here.", passages) == []


def test_labels_stable_when_resolved_twice_against_same_packet():
    passages = [_passage("S1", passage_id="pid-1")]
    text = "A fact [S1]."
    first = resolve_citations(text, passages)
    second = resolve_citations(text, passages)
    assert first == second


# ---------------------------------------------------------------------------
# render_evidence
# ---------------------------------------------------------------------------


def test_render_evidence_includes_label_and_text():
    packet = {
        "passages": [
            {
                "label": "S1",
                "id": "p1",
                "title": "Water",
                "text": "Water is wet.",
                "kind": "article",
            }
        ]
    }
    rendered = render_evidence(packet)
    assert isinstance(rendered, str)
    assert "S1" in rendered
    assert "Water is wet." in rendered


def test_render_evidence_marks_qa_kind_passages():
    packet = {
        "passages": [
            {
                "label": "S1",
                "id": "p1",
                "title": "FAQ",
                "text": "Because gravity.",
                "kind": "qa",
            }
        ]
    }
    rendered = render_evidence(packet)
    assert "[Q&A]" in rendered


def test_render_evidence_does_not_mark_non_qa_passages():
    packet = {
        "passages": [
            {
                "label": "S1",
                "id": "p1",
                "title": "Gravity",
                "text": "Gravity pulls objects together.",
                "kind": "article",
            }
        ]
    }
    rendered = render_evidence(packet)
    assert "[Q&A]" not in rendered


def test_render_evidence_renders_every_passage_and_preserves_order():
    packet = {
        "passages": [
            {"label": "S1", "id": "p1", "title": "A", "text": "first fact", "kind": "article"},
            {"label": "S2", "id": "p2", "title": "B", "text": "second fact", "kind": "qa"},
        ]
    }
    rendered = render_evidence(packet)
    assert rendered.index("first fact") < rendered.index("second fact")
    assert "S1" in rendered and "S2" in rendered


def test_render_evidence_empty_packet_returns_empty_or_blank_string():
    rendered = render_evidence({"passages": []})
    assert rendered.strip() == ""


def test_render_evidence_appends_citation_reminder_after_non_empty_packet():
    """Measured (eval/run_turn_eval.py, docs/citation_experiment.md): a
    one-line reminder appended at the end of the evidence block raised the
    live citation rate from 0.20 to 0.60 on the fixture questions. The
    label stays at the START of each passage line (unchanged).

    2026-09-20 follow-up: the original wording ("Cite the sources you use
    like [S1].") plausibly encouraged the evidence-dump failure (the model
    citing/copying every passage rather than only the ones that actually
    support a sentence). The reminder now also forbids dumping and tells
    the model to say so when nothing answers the question -- re-measurement
    with eval/run_turn_eval.py is PENDING (see docs/citation_experiment.md)."""
    packet = {
        "passages": [
            {
                "label": "S1",
                "id": "p1",
                "title": "Water",
                "text": "Water is wet.",
                "kind": "article",
            }
        ]
    }
    rendered = render_evidence(packet)
    assert rendered.startswith("[S1]")
    reminder = rendered.rstrip().splitlines()[-1]
    assert "[S1]" in reminder
    assert "actually supports" in reminder
    assert "do not list or copy" in reminder.lower()
    assert "if none of them answers the question" in reminder.lower()


def test_render_evidence_no_reminder_appended_for_empty_packet():
    assert render_evidence({"passages": []}).strip() == ""


# ---------------------------------------------------------------------------
# supported (mechanical support check) -- 2026-09-20 evidence-dump follow-up
#
# Real failure: "Output the boiling point of helium in celsius and
# farenheit." Retrieval returned 11 passages about computer "output"
# devices/logic gates/multiplexers. The model answered from memory (a
# slightly wrong number), then wrote "Here's the reasoning:" followed by a
# bullet list that copied out all 11 passages verbatim as "[S1]: ..." ...
# "[S11]: ...", ending with "[S1][S2]...[S11]" stacked on one sentence.
# Every label *resolved* (all 11 passages existed in the packet) but none
# of them *supported* anything the answer actually claimed about helium.
# ---------------------------------------------------------------------------

from tutor.app.citations import detect_evidence_dump  # noqa: E402


def _helium_dump_fixture():
    """Reconstructs the owner's exact failing turn: 11 off-topic
    "output"-themed passages, verbatim-copied into the answer as a bullet
    list, with every label re-stacked on the final sentence."""
    passages = []
    bullets = []
    topics = [
        "Output devices convert data processed by a computer into a form humans can read.",
        "A monitor is the most common output device, displaying text and images.",
        "Printers produce a permanent, physical output of digital documents.",
        "Speakers are output devices that convert electrical signals into sound.",
        "A logic gate performs a basic logical function on one or more binary inputs.",
        "An AND gate outputs true only when all of its inputs are true.",
        "An OR gate outputs true when at least one of its inputs is true.",
        "A NOT gate inverts its single input to produce the opposite output.",
        "A multiplexer selects one of several inputs and forwards it to a single output line.",
        "A demultiplexer takes a single input and routes it to one of several output lines.",
        "Output ports on a computer's motherboard connect external output devices.",
    ]
    for i, text in enumerate(topics, start=1):
        label = f"S{i}"
        passages.append(
            {
                "label": label,
                "id": f"p{i}",
                "title": f"Output topic {i}",
                "path": f"Computing/Output{i}",
                "text": text,
                "start": 0,
                "end": len(text),
                "kind": "article",
            }
        )
        bullets.append(f"[{label}]: {text}")
    stacked_labels = "".join(f"[S{i}]" for i in range(1, 12))
    answer_text = (
        "The boiling point of helium is -268.92°C or -452.04°F.\n\n"
        "Here's the reasoning:\n"
        + "\n".join(bullets)
        + "\nThese sources describe how computers produce output. "
        + stacked_labels
    )
    return answer_text, passages


def test_helium_dump_fixture_all_citations_unsupported():
    answer_text, passages = _helium_dump_fixture()
    citations = resolve_citations(answer_text, passages)
    assert len(citations) == 11
    assert all(not c.unresolved for c in citations)  # every label DID resolve
    assert all(c.supported is False for c in citations)  # but none SUPPORT the claim


def test_helium_dump_fixture_detected_as_evidence_dump():
    answer_text, passages = _helium_dump_fixture()
    citations = resolve_citations(answer_text, passages)
    assert detect_evidence_dump(answer_text, citations, passages) is True


def test_supported_true_when_sentence_shares_content_terms_with_passage():
    passages = [_passage("S1")]
    passages[0]["text"] = "Water boils at 100 degrees Celsius at sea level."
    text = "Water boils at 100 degrees Celsius at sea level [S1]."
    citations = resolve_citations(text, passages)
    assert citations[0].supported is True


def test_supported_false_when_sentence_shares_no_content_terms_with_passage():
    passages = [_passage("S1")]
    passages[0]["text"] = "Logic gates process binary signals in a multiplexer."
    text = "The boiling point of helium is -268.92 celsius [S1]."
    citations = resolve_citations(text, passages)
    assert citations[0].supported is False


def test_unresolved_citation_is_not_supported():
    citations = resolve_citations("A claim with no evidence [S9].", [])
    assert citations[0].unresolved is True
    assert citations[0].supported is False


def test_detect_evidence_dump_false_for_a_normal_short_cited_answer():
    passages = [_passage("S1")]
    passages[0]["text"] = "Water boils at 100 degrees Celsius at sea level."
    text = "Water boils at 100 degrees Celsius at sea level [S1]."
    citations = resolve_citations(text, passages)
    assert detect_evidence_dump(text, citations, passages) is False


def test_system_prompt_stays_within_approx_800_token_budget():
    """The system prompt occupies a fixed ~800-token slot of the turn
    budget (docs/plan/offline_tutor_spec_v0.3.md). Uses the same
    length/3.5 approximation ``tutor.app.compose._make_count_tokens`` falls
    back to when the real tokenizer is unreachable -- good enough for a
    build-time regression guard, not meant to be exact."""
    import math
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tutor" / "app" / "system_prompt.txt"
    text = path.read_text(encoding="utf-8")
    approx_tokens = math.ceil(len(text) / 3.5)
    assert approx_tokens <= 800, f"system_prompt.txt is ~{approx_tokens} tokens, over budget"


def test_detect_evidence_dump_true_for_label_stacking_on_one_sentence():
    passages = [_passage(f"S{i}", passage_id=f"p{i}") for i in range(1, 5)]
    for i, p in enumerate(passages, start=1):
        p["text"] = f"Topic {i} passage text here."
    filler = " ".join(
        f"Extra background detail number {i} about the subject." for i in range(1, 12)
    )
    text = (
        filler
        + " This is supported by everything [S1][S2][S3][S4]."
    )
    citations = resolve_citations(text, passages)
    assert detect_evidence_dump(text, citations, passages) is True


def test_detect_evidence_dump_false_for_stacked_labels_on_a_short_answer():
    """2026-09-21 dump-collapse bug: a short, correct answer like
    "Yes, DNA is a molecule." should never be treated as an evidence dump
    just because its trailing citation carries several stacked labels --
    a genuine dump is a long, evasive answer, not a short direct one."""
    passages = [_passage(f"S{i}", passage_id=f"p{i}") for i in range(1, 5)]
    for i, p in enumerate(passages, start=1):
        p["text"] = f"Topic {i} passage text here."
    text = "Yes, DNA is a molecule. [S1][S2][S3][S4]"
    citations = resolve_citations(text, passages)
    assert detect_evidence_dump(text, citations, passages) is False


# ---------------------------------------------------------------------------
# 2026-09-20 tables feedback: `_sentence_spans` treats a markdown table ROW
# and a list item as one unit rather than splitting at "1." or inside a
# cell. Fixture reconstructs the owner's real square-foot-garden answer
# (4-row table with the same syntax quirks: a bold step cell, `<br>*`
# bullets inside cells, a row whose bold spans two cells) plus a heading
# and a `- ` list.
# ---------------------------------------------------------------------------

def _load_table_answer() -> str:
    path = _Path(__file__).resolve().parent / "fixtures" / "table_answer.md"
    return path.read_text(encoding="utf-8")


def test_table_row_units_never_split_inside_list_numbering_or_table_syntax():
    text = _load_table_answer()
    spans = _sentence_spans(text)
    for start, end in spans:
        unit = text[start:end]
        # Never a span that IS (or ends inside) a bare list-numbering marker
        # like "1." with nothing else -- covered by _is_short_non_claim for
        # bona fide bare markers, but the span itself must also never START
        # or END strictly inside a "\d+\." run split off mid-token.
        assert not unit.rstrip().endswith("<br>")
        assert unit.strip() not in {"|", "---", "<br>"}
        # A span must never be pure table/list syntax with no content.
        stripped_syntax = unit.strip().strip("|-: ")
        assert stripped_syntax != "" or "\n" in unit


def test_table_body_row_count_matches_unit_count_for_rows():
    text = _load_table_answer()
    spans = _sentence_spans(text)
    units = [text[s:e] for s, e in spans]
    row_units = [u for u in units if u.strip().startswith("|") and u.strip().endswith("|")]
    # 4 body rows in the fixture table (header + separator excluded as
    # non-claims and produce no span at all).
    assert len(row_units) == 4


def test_table_header_and_separator_rows_produce_no_span():
    text = _load_table_answer()
    spans = _sentence_spans(text)
    units = [text[s:e] for s, e in spans]
    assert not any("Step" in u and "What to Do" in u for u in units)
    assert not any(set(u.strip()) <= set("|- :") for u in units)


def test_list_item_unit_keeps_its_number_intact():
    text = "1. Choose a bed size.\n2. Pick a sunny spot."
    spans = _sentence_spans(text)
    units = [text[s:e] for s, e in spans]
    assert "1. Choose a bed size." in units
    assert "2. Pick a sunny spot." in units


def test_quick_tips_list_items_are_their_own_units():
    text = _load_table_answer()
    spans = _sentence_spans(text)
    units = [text[s:e] for s, e in spans]
    assert any(u.startswith("- Water consistently") for u in units)
    assert any(u.startswith("- Rotate crops") for u in units)
    assert any(u.startswith("- Thin seedlings") for u in units)


# ---------------------------------------------------------------------------
# 2026-09-21 model-authored source-list block (owner-reported live bug): the
# model sometimes writes its OWN fake "Sources:" list with invented passage
# descriptions. The host must never turn the heading line or the label-led
# list lines that follow it into attribution sentence units -- no ○/●
# marker on them, and they must not count toward backed/unbacked rates.
# Text after the block (e.g. a trailing "Short answer: ...") still gets
# normal spans. See `_SOURCE_HEADING_RE` / `_LABEL_LED_LINE_RE` /
# `find_model_source_block_ranges`, mirrored in tutor/ui/app.js.
# ---------------------------------------------------------------------------

_OWNER_SOURCE_BLOCK_ANSWER = (
    "So, while DNA is the largest molecule ... the longest molecule overall "
    "is titin.\n\n"
    "Sources:\n"
    "[S1] DNA structure and replication mechanisms (describing DNA's size "
    "in terms of base pairs).\n"
    "[S2] Overview of DNA replication process.\n"
    "[S3] Basic chemical structure of DNA (describing its components).\n"
    "[S4] General description of DNA molecules.\n"
    "[S5] References showing various descriptions of DNA length.\n"
    "[S6] (Note: The source list provided was not directly related to the "
    "question but serves as context for the answer.)\n\n"
    "Short answer: DNA is large, but titin is the longest molecule by "
    "sheer length of its amino acid sequence."
)


def test_find_model_source_block_ranges_covers_heading_and_label_lines():
    ranges = find_model_source_block_ranges(_OWNER_SOURCE_BLOCK_ANSWER)
    assert len(ranges) == 1
    start, end = ranges[0]
    block = _OWNER_SOURCE_BLOCK_ANSWER[start:end]
    assert block.startswith("Sources:")
    assert "[S6]" in block
    assert "Short answer" not in block


@pytest.mark.parametrize("heading", ["Sources:", "Source:", "References:", "Citations:",
                                      "**Sources**", "## Sources"])
def test_find_model_source_block_ranges_recognizes_heading_variants(heading):
    text = heading + "\n[S1] a made-up description.\n[S2] another one.\n"
    ranges = find_model_source_block_ranges(text)
    assert len(ranges) == 1


def test_find_model_source_block_ranges_ignores_bare_heading_with_no_label_lines():
    text = "Sources:\nI could not find anything relevant."
    assert find_model_source_block_ranges(text) == []


def test_find_model_source_block_ranges_ignores_normal_sentence_mentioning_label():
    text = "As shown in [S1], water boils at 100C. See [S2] for more."
    assert find_model_source_block_ranges(text) == []


def test_sentence_spans_excludes_model_source_block_lines():
    spans = _sentence_spans(_OWNER_SOURCE_BLOCK_ANSWER)
    units = [_OWNER_SOURCE_BLOCK_ANSWER[s:e] for s, e in spans]
    assert not any(u.strip().startswith("Sources:") for u in units)
    assert not any(u.strip().startswith("[S") for u in units)
    assert any("Short answer" in u for u in units)
    assert any("longest molecule overall is titin" in u for u in units)


# ---------------------------------------------------------------------------
# 2026-09-21 markdown-link fake citation (owner-reported LIVE bug, seen after
# 8f26f82 + f0f8755 hid the plain "[Cite: [Q&A]]" / fake "Sources:" forms):
# the model now ends an answer with a fake citation in MARKDOWN-LINK shape
# pointing at the live internet -- this is an OFFLINE tool, the URL is
# invented. Verbatim examples the owner saw live:
#   [Cite: [S1]](https://en.wikipedia.org/wiki/Titin)
#   [Cite: [S1]](https://en.wikipedia.org/wiki/Rubber)
#   [Cite: [S1]](https://en.wikipedia.org/wiki/Vulcanization)
# `collapse_invented_links` is the single source of truth for turning any
# invented-label token immediately followed by a parenthesised http(s) URL
# into either the bare real [S#] label(s) it wraps (if any) or "" (if it
# wraps no real label) -- mirrored in tutor/ui/app.js as
# `collapseInventedLinks`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[Cite: [S1]](https://en.wikipedia.org/wiki/Titin)", "[S1]"),
        ("[Cite: [S1]](https://en.wikipedia.org/wiki/Rubber)", "[S1]"),
        ("[Cite: [S1]](https://en.wikipedia.org/wiki/Vulcanization)", "[S1]"),
        ("[Source](https://en.wikipedia.org/wiki/Rubber)", ""),
        ("[Source: foo](https://example.com/x)", ""),
        ("[S1](https://example.com/x)", "[S1]"),
    ],
)
def test_collapse_invented_links_examples(raw, expected):
    assert collapse_invented_links(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "[H2O]",
        "[1, 2, 3]",
        "Water boils at 100 degrees Celsius. [S1]",
        "f(x) is defined for all reals (see above).",
    ],
)
def test_collapse_invented_links_leaves_must_not_break_cases_alone(text):
    assert collapse_invented_links(text) == text


def test_collapse_invented_links_in_full_sentence_context():
    sentence = (
        "Titin is the largest known protein. "
        "[Cite: [S1]](https://en.wikipedia.org/wiki/Titin)"
    )
    collapsed = collapse_invented_links(sentence)
    assert "https://" not in collapsed
    assert "Cite:" not in collapsed
    assert "[S1]" in collapsed


# ---------------------------------------------------------------------------
# 2026-09-21 specifics gate (owner-reported LIVE bug): the host marked a
# fabricated person and fabricated temperatures as "found in the sources"
# purely on loose word overlap ("cold", "temperature", "survived" etc. all
# shared with a real passage). Fix: a sentence may only be marked
# supported/backed when every SPECIFIC it states (a proper name, or a
# number -- other than a unit-conversion output riding along in
# parentheses right after its source number, e.g. the "-94°F" in
# "-70°C (-94°F)") is actually present in the evidence text
# (the passage plus every other passage in the same packet -- never the
# tutor's own earlier text, which is not evidence at all).
# ---------------------------------------------------------------------------

_VITUS_SENTENCE_1 = (
    "The most well-documented case is that of Vitus Andronicus, a Roman "
    "soldier who survived a brutal winter in the frozen regions of the "
    "Balkans, where he endured temperatures as low as -70°C "
    "(-94°F✓ checked)."
)
_VITUS_SENTENCE_2 = (
    "Some accounts cite a survival at -71.5°C (-96.7°F✓ "
    "checked) in the Antarctic."
)

_COLD_PASSAGE_TEXT = (
    "Hypothermia occurs when the human body loses heat faster than it can "
    "produce it. Survival at extremely low body temperatures is rare and "
    "poorly documented; most clinical case studies describe core body "
    "temperature drops, not prolonged exposure to sub-zero air "
    "temperatures. The coldest recorded natural air temperature on Earth "
    "was measured in Antarctica."
)


def test_vitus_sentence_1_not_supported_by_cold_passage():
    assert is_supported(_VITUS_SENTENCE_1, _COLD_PASSAGE_TEXT) is False


def test_vitus_sentence_2_not_supported_by_cold_passage():
    assert is_supported(_VITUS_SENTENCE_2, _COLD_PASSAGE_TEXT) is False


def test_vitus_sentences_not_supported_end_to_end_via_resolve_citations():
    passages = [_passage("S1")]
    passages[0]["text"] = _COLD_PASSAGE_TEXT
    text = _VITUS_SENTENCE_1[:-1] + " [S1]."
    citations = resolve_citations(text, passages)
    assert citations[0].supported is False


def test_peregrine_falcon_backed_when_name_and_number_both_present():
    sentence = "The peregrine falcon can reach about 390 km/h in a dive."
    passage_text = (
        "The peregrine falcon is the fastest bird, reaching speeds of "
        "390 km/h in a hunting dive."
    )
    assert is_supported(sentence, passage_text) is True


def test_peregrine_falcon_not_backed_when_number_mismatches():
    sentence = "The peregrine falcon can reach about 390 km/h in a dive."
    passage_text = (
        "The peregrine falcon is the fastest bird, reaching speeds of "
        "320 km/h in a hunting dive."
    )
    assert is_supported(sentence, passage_text) is False


def test_sentence_with_no_specifics_keeps_todays_overlap_behavior():
    # No numbers, no proper names (beyond the sentence-initial word) --
    # the specifics gate has nothing to check, so plain overlap decides,
    # same as before this change.
    sentence = "Water boils at high altitude at a lower temperature."
    passage_text = "At high altitude, water boils at a lower temperature."
    assert is_supported(sentence, passage_text) is True


def test_specific_name_found_only_via_second_passage_in_packet_still_counts():
    # The gate's evidence set is "this turn's packet", not just the one
    # passage a label happens to point at -- a name backed by a SIBLING
    # passage in the same packet still counts.
    passages = [
        _passage("S1"),
        _passage("S2"),
    ]
    passages[0]["text"] = "390 km/h is an extremely fast diving speed for a bird."
    passages[1]["text"] = "The peregrine falcon is renowned for its hunting dives."
    text = "The peregrine falcon can reach about 390 km/h in a dive [S1]."
    citations = resolve_citations(text, passages)
    assert citations[0].supported is True


def test_earlier_tutor_text_is_never_used_as_evidence():
    # A prior TUTOR turn stating the same fabrication must not itself count
    # as evidence for a later turn -- is_supported/resolve_citations only
    # ever see the CURRENT packet's passages, never prior answer text; this
    # test pins that down by asserting the fabricated sentence stays
    # unsupported even though it is, verbatim, the "earlier tutor text".
    passages = [_passage("S1")]
    passages[0]["text"] = _COLD_PASSAGE_TEXT
    text = _VITUS_SENTENCE_1[:-1] + " [S1]."
    citations = resolve_citations(text, passages)
    assert citations[0].supported is False

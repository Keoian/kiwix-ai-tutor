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

from tutor.app.citations import Citation, extract_labels, render_evidence, resolve_citations


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

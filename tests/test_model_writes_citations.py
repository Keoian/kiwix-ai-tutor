"""Tests for ``app.model_writes_citations`` (default False, owner decision
2026-09-21 -- see docs/attribution_design.md, "Model-written labels are no
longer requested"). The host's own per-sentence source dots already link
each sentence to the passage that backs it, independent of whether the
model itself writes an ``[S#]`` label, so the system prompt no longer asks
the model to write one by default, and the evidence-tail reminder drops
the "cite ... like [S1]" instruction while keeping the other two parts.
``True`` must reproduce today's bytes exactly.
"""

from tutor.app.agent_loop import _CITE_LABEL_SECTION, build_system_text
from tutor.app.citations import (
    _CITATION_REMINDER,
    _NO_LABEL_REMINDER,
    citation_reminder_text,
    render_evidence,
)


def _passage(label="S1", passage_id="p1", text="Water boils at 100C."):
    return {"id": passage_id, "label": label, "text": text}


def test_default_system_prompt_omits_citation_label_instruction():
    system_text = build_system_text()
    assert "numbered label given to you" not in system_text
    assert _CITE_LABEL_SECTION.strip() not in system_text


def test_model_writes_citations_true_restores_the_instruction():
    system_text = build_system_text(model_writes_citations=True)
    assert "numbered label given to you" in system_text
    assert _CITE_LABEL_SECTION.strip() in system_text


def test_model_writes_citations_true_still_has_no_specifics_and_child_safe():
    on = build_system_text(model_writes_citations=True)
    off = build_system_text(model_writes_citations=False)
    # Turning the citation-label instruction on/off must not touch the
    # other default-on sections.
    assert "Names and numbers must come from the library" in on
    assert "Names and numbers must come from the library" in off
    assert "Questions about bodies, sex and growing up" in on
    assert "Questions about bodies, sex and growing up" in off


def test_citation_reminder_text_true_is_old_bytes():
    assert citation_reminder_text(True) == _CITATION_REMINDER


def test_citation_reminder_text_false_drops_only_the_label_instruction():
    text = citation_reminder_text(False)
    assert text == _NO_LABEL_REMINDER
    assert "like [S1]" not in text
    assert "Do not list or copy the sources" in text
    assert "say so" in text


def test_render_evidence_default_omits_label_instruction():
    packet = {"passages": [_passage()]}
    rendered = render_evidence(packet, model_writes_citations=False)
    assert "like [S1]" not in rendered
    assert "Do not list or copy the sources" in rendered
    assert "[S1]" in rendered  # the passage is still labelled


def test_render_evidence_true_reproduces_old_bytes():
    packet = {"passages": [_passage()]}
    default_call = render_evidence(packet)
    explicit_true = render_evidence(packet, model_writes_citations=True)
    assert default_call == explicit_true
    assert "like [S1]" in default_call

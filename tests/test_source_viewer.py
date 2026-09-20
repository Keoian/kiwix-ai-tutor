"""RED tests for tutor.app.source_view.

Authoritative source: docs/plan/offline_tutor_spec_v0.3.md §12 ("Source
viewer | Saved extract, cited sentence highlighted...") and §11 (source
viewer highlights the exact sentence(s) cited, using stored character
offsets).

Contract decisions made here (spec silent on exact function shapes):

- ``build_source_view(snapshot, *, context_chars=600) -> SourceView`` takes
  a ``tutor.retrieval.snapshots.Snapshot`` and returns a ``SourceView``
  dataclass with fields: ``title``, ``path``, ``archive_id``,
  ``heading_path`` (tuple[str, ...]), ``text`` (the full snapshot text),
  ``highlight`` (a ``(start, end)`` tuple, exact cited span within
  ``text``), ``context_before``/``context_after`` (surrounding text,
  bounded by ``context_chars``, snapped to sentence boundaries), and
  ``kind``.
- ``sentence_span(text, start, end) -> (s, e)`` expands ``(start, end)`` out
  to the nearest sentence boundaries (``.``, ``!``, ``?`` followed by
  whitespace/EOF, or a leading capital/start-of-text boundary) without ever
  crossing past the edges of ``text`` itself, and never narrows past the
  original ``(start, end)``.
- The API never returns raw archive HTML: ``SourceView.text`` (and the
  context fields) must never contain a literal ``"<script"`` substring, even
  when the underlying snapshot text was not fully sanitized upstream --
  escaping for on-page rendering is the UI's job, but the source_view layer
  strips/refuses obviously unsafe markup rather than passing it through
  verbatim.
"""

from __future__ import annotations

from tutor.app.source_view import SourceView, build_source_view, sentence_span


def _snapshot(
    text="Water is a molecule. It boils at 100C at sea level. Ice floats on water.",
    start=21,
    end=51,
    heading_path=("Water", "Properties"),
    kind="article",
):
    from types import SimpleNamespace

    return SimpleNamespace(
        passage_id="p1",
        archive_id="wiki_demo",
        path="A/Water",
        title="Water",
        heading_path=heading_path,
        start=start,
        end=end,
        text=text,
        fingerprint_digest="deadbeef",
        created_at="2026-01-01T00:00:00+00:00",
        kind=kind,
    )


# ---------------------------------------------------------------------------
# build_source_view
# ---------------------------------------------------------------------------


def test_highlight_span_is_exact():
    snap = _snapshot()
    view = build_source_view(snap)
    assert isinstance(view, SourceView)
    s, e = view.highlight
    assert view.text[s:e] == "It boils at 100C at sea level."


def test_basic_fields_copied_through():
    snap = _snapshot()
    view = build_source_view(snap)
    assert view.title == "Water"
    assert view.path == "A/Water"
    assert view.archive_id == "wiki_demo"
    assert view.heading_path == ("Water", "Properties")
    assert view.kind == "article"


def test_context_before_and_after_present_and_bounded():
    long_text = ("Sentence number %d. " * 200) % tuple(range(200))
    start = long_text.index("Sentence number 100.")
    end = start + len("Sentence number 100.")
    snap = _snapshot(text=long_text, start=start, end=end)
    view = build_source_view(snap, context_chars=50)
    assert len(view.context_before) <= 80  # bounded, with slack for sentence snapping
    assert len(view.context_after) <= 80
    assert view.text[view.highlight[0] : view.highlight[1]] == "Sentence number 100."


def test_no_script_tag_ever_returned():
    snap = _snapshot(
        text="Before text. <script>alert(1)</script> It boils at 100C. After text.",
        start=40,
        end=58,
    )
    view = build_source_view(snap)
    assert "<script" not in view.text
    assert "<script" not in view.context_before
    assert "<script" not in view.context_after


def test_unicode_safe_highlight():
    text = "Café résumé. Wassér boils. Ice floats on water."
    start = text.index("Wassér boils.")
    end = start + len("Wassér boils.")
    snap = _snapshot(text=text, start=start, end=end)
    view = build_source_view(snap)
    s, e = view.highlight
    assert view.text[s:e] == "Wassér boils."


def test_kind_qa_passed_through():
    snap = _snapshot(kind="qa")
    view = build_source_view(snap)
    assert view.kind == "qa"


# ---------------------------------------------------------------------------
# sentence_span
# ---------------------------------------------------------------------------


def test_sentence_span_snaps_to_full_sentence():
    text = "First sentence here. Second sentence is cited. Third sentence trails."
    inner_start = text.index("Second")
    inner_end = inner_start + len("Second sentence is")  # mid-sentence, not full
    s, e = sentence_span(text, inner_start, inner_end)
    assert text[s:e].strip() == "Second sentence is cited."


def test_sentence_span_never_narrower_than_input():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    start = text.index("Delta")
    end = start + len("Delta epsilon zeta.")
    s, e = sentence_span(text, start, end)
    assert s <= start
    assert e >= end


def test_sentence_span_never_crosses_text_edges():
    text = "Only one sentence in this whole passage."
    s, e = sentence_span(text, 5, 8)
    assert s >= 0
    assert e <= len(text)


def test_sentence_span_at_start_of_text():
    text = "Start sentence here. Followed by another one."
    s, e = sentence_span(text, 0, 5)
    assert s == 0
    assert text[s:e].strip().startswith("Start sentence here.")


def test_sentence_span_unicode_safe():
    text = "Héllo wörld. Café is nice. Zürich is a city."
    start = text.index("Café")
    end = start + len("Café is nice")
    s, e = sentence_span(text, start, end)
    assert text[s:e].strip() == "Café is nice."

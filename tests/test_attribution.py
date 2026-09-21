"""RED tests for `tutor.app.citations.attribute_sentences` (does not exist
yet -- ImportError expected, this is the failing-first-test commit).

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §14 (three kinds
of statement: source-backed / computed / tutor's own example) and §11
("[S#] labels resolved by the host and rejected if unresolved", "the source
viewer highlights the exact sentence(s) the answer cited, using the
passage's stored character offsets"). See docs/attribution_design.md.

Granite's measured defect (docs/citation_experiment.md,
data/granite_soak10_v2.turns.json): it staples a single trailing [S1] to a
multi-sentence paragraph instead of the claim it actually backs. This module
lets the host attribute the earlier, unlabeled sentences to the same passage
by term overlap, without ever rewriting the stored answer text or fabricating
a citation the model did not write (per §11).

Contract decisions made here (spec silent):
- `attribute_sentences(answer, passages) -> AttributionResult` where
  `passages` is the same `packet_passages` shape `resolve_citations` takes
  (list of dicts with "label"/"id"/"text"/... keys, see `_passage()` in
  tests/test_citations.py).
- `AttributionResult.attributions`: list of items, each with
  `sentence_span` (a `(start, end)` char-offset tuple into the *original,
  unmodified* `answer` string -- `answer[start:end] == sentence text`),
  `passage_id`, `label`, `score` (float, higher is a better match),
  `model_cited` (True iff the sentence itself carries that passage's [S#]
  label).
- `AttributionResult.unbacked_spans`: list of items, each with `span`
  (same offset convention) and `reason` in {"unbacked", "unbacked_number"}.
  "unbacked_number" is used when the sentence contains a figure (a number
  token) that appears in no passage at all, even if some of the sentence's
  words overlap a passage.
- A short non-claim (a question to the student, an exclamation like "Great
  question!") lands in neither list.
- Numbers count as terms for overlap purposes (the shared `tokenize` already
  keeps digit runs); "-268.93" and "−268.93" (ASCII hyphen-minus vs.
  Unicode minus) are treated as the same figure.
"""

from __future__ import annotations

import json
from pathlib import Path

from tutor.app.citations import attribute_sentences

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "granite_soak10_v2.turns.json"


def _passage(label, passage_id, text, title="Title", path="A/Title", kind="article"):
    return {
        "label": label,
        "id": passage_id,
        "title": title,
        "path": path,
        "text": text,
        "start": 0,
        "end": len(text),
        "kind": kind,
    }


def _load_granite_paragraph() -> str:
    """The exact 5-sentence, single-trailing-[S1] paragraph from the
    owner's Granite soak, if the fixture file is present; otherwise a
    realistic stand-in with the same shape."""
    if _DATA_FILE.exists():
        turns = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
        for turn in turns:
            text = turn.get("answer_text", "")
            if text.count("[S1]") == 1 and text.rstrip().endswith("[S1]") and text.count(".") >= 4:
                return text
    return (
        "Photosynthesis is the process by which green plants use sunlight "
        "to make food. During photosynthesis, plants convert light energy "
        "into chemical energy stored in glucose. This process turns carbon "
        "dioxide and water into glucose and oxygen. The glucose produced "
        "serves as an energy source for the plant. Many students enjoy "
        "gardening as a hobby. [S1]"
    )


# ---------------------------------------------------------------------------
# (a) Granite shape: 5-sentence paragraph, one trailing [S1]
# ---------------------------------------------------------------------------


def test_granite_shape_unlabeled_related_sentences_attributed_to_trailing_citation():
    answer = _load_granite_paragraph()
    passage_text = (
        "Photosynthesis is the process green plants use to convert light "
        "energy into chemical energy, producing glucose from carbon dioxide "
        "and water; the glucose is used by the plant as an energy source."
    )
    passages = [_passage("S1", "p1", passage_text)]

    result = attribute_sentences(answer, passages)

    attributed_texts = [
        answer[a.sentence_span[0] : a.sentence_span[1]] for a in result.attributions
    ]
    joined = " ".join(attributed_texts)
    assert "Photosynthesis is the process" in joined
    assert "chemical energy" in joined
    for a in result.attributions:
        assert a.passage_id == "p1"
        assert a.label == "S1"

    # exactly one attribution actually carries the [S1] label itself.
    cited_flags = [a.model_cited for a in result.attributions]
    assert cited_flags.count(True) == 1
    assert cited_flags.count(False) >= 1

    # the unrelated sentence (hobby/gardening, if present) is not attributed.
    unbacked_texts = [answer[u.span[0] : u.span[1]] for u in result.unbacked_spans]
    if "gardening" in answer:
        assert any("gardening" in t for t in unbacked_texts)


# ---------------------------------------------------------------------------
# (b) reuse the owner's 11-passage evidence-dump fixture
# ---------------------------------------------------------------------------


def _helium_dump_fixture():
    """Mirrors tests/test_citations.py::_helium_dump_fixture exactly (11
    off-topic "output"-themed passages, verbatim-copied into the answer,
    every label stacked on the final sentence)."""
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
            _passage(label, f"p{i}", text, title=f"Output topic {i}", path=f"Computing/Output{i}")
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


def test_helium_dump_fixture_first_sentence_unbacked_number():
    answer, passages = _helium_dump_fixture()
    result = attribute_sentences(answer, passages)

    first_sentence_end = answer.index("\n")
    reasons = {
        u.reason
        for u in result.unbacked_spans
        if u.span[0] < first_sentence_end and "helium" in answer[u.span[0] : u.span[1]]
    }
    assert "unbacked_number" in reasons

    # no passage mentions helium/boiling point, so nothing should attribute
    # the opening sentence to any of the 11 off-topic passages.
    for a in result.attributions:
        assert "helium" not in answer[a.sentence_span[0] : a.sentence_span[1]]


# ---------------------------------------------------------------------------
# (c) best passage wins when two passages overlap
# ---------------------------------------------------------------------------


def test_best_overlapping_passage_wins():
    answer = "The mitochondria is the powerhouse of the cell, producing ATP through respiration."
    weak = _passage("S1", "p1", "Cells have many organelles including the nucleus and ribosomes.")
    strong = _passage(
        "S2",
        "p2",
        "The mitochondria is the powerhouse of the cell; it produces ATP via cellular respiration.",
    )
    result = attribute_sentences(answer, [weak, strong])

    assert len(result.attributions) == 1
    assert result.attributions[0].passage_id == "p2"
    assert result.attributions[0].label == "S2"


# ---------------------------------------------------------------------------
# (d) minimum-overlap threshold
# ---------------------------------------------------------------------------


def test_stopword_only_overlap_is_unbacked():
    answer = "The quick brown fox jumps over the lazy dog."
    passage = _passage("S1", "p1", "It is what it is, and that is that for the most part.")
    result = attribute_sentences(answer, [passage])

    assert result.attributions == []
    assert len(result.unbacked_spans) == 1
    assert result.unbacked_spans[0].reason == "unbacked"


def test_single_shared_term_below_threshold_is_unbacked():
    answer = "Gardening is a relaxing hobby enjoyed by many retirees worldwide."
    passage = _passage(
        "S1", "p1", "Photosynthesis lets plants make food using sunlight, water, and carbon dioxide"
    )
    # only "plants"/"food"-style incidental overlap at most, well under both
    # the absolute-count and fractional thresholds.
    result = attribute_sentences(answer, [passage])
    assert result.attributions == []
    assert len(result.unbacked_spans) == 1


# ---------------------------------------------------------------------------
# (e) figures: appears in no passage -> unbacked_number; appears in a
# passage -> attributed. ASCII hyphen-minus and Unicode minus treated alike.
# ---------------------------------------------------------------------------


def test_figure_absent_from_every_passage_is_unbacked_number():
    answer = "The temperature dropped to -268.93 degrees during the experiment."
    passage = _passage("S1", "p1", "Experiments require careful temperature control in the lab.")
    result = attribute_sentences(answer, [passage])

    assert result.attributions == []
    assert len(result.unbacked_spans) == 1
    assert result.unbacked_spans[0].reason == "unbacked_number"


def test_label_digits_are_not_flagged_as_unbacked_figures():
    # docs/soak_v3_analysis.md §6: a sentence's own trailing [S18]-style
    # label was being scanned as if "18" were a claimed figure, flagging
    # it unbacked_number even though no real number was ever asserted.
    answer = "This method is a fundamental skill in working with fractions [S18]."
    passage = _passage("S1", "p1", "Fractions require finding a common denominator.")
    result = attribute_sentences(answer, [passage])

    assert all(u.reason != "unbacked_number" for u in result.unbacked_spans)


def test_leading_list_number_is_not_flagged_as_unbacked_figure():
    answer = "1. Multiply the numerator and denominator by the same nonzero value."
    passage = _passage("S1", "p1", "Fractions require finding a common denominator.")
    result = attribute_sentences(answer, [passage])

    assert all(u.reason != "unbacked_number" for u in result.unbacked_spans)


def test_figure_present_in_passage_with_unicode_minus_is_attributed():
    answer = "The boiling point of helium is -268.93 degrees Celsius."
    # passage uses the Unicode minus sign (−), answer uses ASCII hyphen.
    passage = _passage("S1", "p1", "Helium boils at −268.93 degrees Celsius at standard pressure.")
    result = attribute_sentences(answer, [passage])

    assert len(result.attributions) == 1
    assert result.attributions[0].passage_id == "p1"
    assert result.unbacked_spans == []


# ---------------------------------------------------------------------------
# (f) spans index the ORIGINAL answer exactly; labels/markdown don't break
# them; the answer is never modified.
# ---------------------------------------------------------------------------


def test_spans_index_original_answer_with_labels_and_markdown_intact():
    answer = "**Water** boils at 100 degrees Celsius at sea level [S1]. This is a fun fact!"
    passage = _passage(
        "S1", "p1", "Water boils at 100 degrees Celsius at standard atmospheric pressure."
    )
    before = str(answer)

    result = attribute_sentences(answer, [passage])

    assert answer == before  # never modified
    for a in result.attributions:
        start, end = a.sentence_span
        assert answer[start:end] == answer[start:end]  # offsets are valid slices
        assert 0 <= start < end <= len(answer)
    for u in result.unbacked_spans:
        start, end = u.span
        assert 0 <= start < end <= len(answer)
    # the cited sentence's slice should contain the [S1] label and the bold
    # markers untouched.
    cited = [a for a in result.attributions if a.model_cited]
    assert len(cited) == 1
    start, end = cited[0].sentence_span
    assert "[S1]" in answer[start:end]
    assert "**Water**" in answer[start:end] or answer[start:end].startswith("Water")


# ---------------------------------------------------------------------------
# (g) empty answer / no passages -> everything unbacked (or nothing), no
# crash.
# ---------------------------------------------------------------------------


def test_empty_answer_no_crash():
    result = attribute_sentences("", [])
    assert result.attributions == []
    assert result.unbacked_spans == []


def test_no_passages_everything_unbacked():
    answer = "Water boils at 100 degrees Celsius at sea level. Salt lowers that boiling point."
    result = attribute_sentences(answer, [])
    assert result.attributions == []
    assert len(result.unbacked_spans) == 2
    for u in result.unbacked_spans:
        assert u.reason == "unbacked"


# ---------------------------------------------------------------------------
# (h) short non-claims (questions to the student, exclamations) are neither
# attributed nor flagged.
# ---------------------------------------------------------------------------


def test_load_granite_paragraph_falls_back_when_data_file_is_absent(monkeypatch, tmp_path):
    """data/granite_soak10_v2.turns.json is gitignored -- it won't exist on
    a clean checkout or CI (windows + ubuntu). Confirm the "when present,
    else built-in" fallback in ``_load_granite_paragraph`` really takes
    over then, without moving/deleting anything in data/: point the module
    constant at a path that never exists."""
    import tests.test_attribution as this_module

    monkeypatch.setattr(this_module, "_DATA_FILE", tmp_path / "does-not-exist.turns.json")
    text = this_module._load_granite_paragraph()
    assert text.startswith("Photosynthesis is the process")
    assert text.rstrip().endswith("[S1]")


def test_short_non_claims_are_ignored_entirely():
    answer = (
        "Water boils at 100 degrees Celsius at sea level [S1]. Great question! "
        "What do you think happens at altitude?"
    )
    passage = _passage(
        "S1", "p1", "Water boils at 100 degrees Celsius at standard atmospheric pressure."
    )
    result = attribute_sentences(answer, [passage])

    all_spans = [a.sentence_span for a in result.attributions] + [
        u.span for u in result.unbacked_spans
    ]
    covered = "".join(answer[s:e] for s, e in all_spans)
    assert "Great question" not in covered
    assert "What do you think" not in covered

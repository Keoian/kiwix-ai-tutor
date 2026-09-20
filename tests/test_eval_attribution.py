"""RED-first tests for ``eval.attribution_scoring`` (does not exist yet at
commit time) and the ``backed_sentence_rate`` / ``unbacked_number_rate``
metrics it feeds into ``eval/run_turn_eval.py`` and
``eval/run_lesson_soak.py``. Fakes only, no live server, no held-out split.
"""

from __future__ import annotations

from eval.attribution_scoring import (
    micro_average_backed_sentence_rate,
    sentence_attribution_counts,
    sentence_attribution_counts_from_event,
    unbacked_number_rate,
)


def _passage(label, passage_id, text):
    return {"label": label, "id": passage_id, "title": "T", "path": "A/T", "text": text}


def test_counts_all_attributed_when_sentence_matches_passage():
    passage = _passage(
        "S1", "p1", "The mitochondria is the powerhouse of the cell, producing ATP."
    )
    answer = "The mitochondria is the powerhouse of the cell [S1]."
    counts = sentence_attribution_counts(answer, [passage])
    assert counts["attributed_sentences"] == 1
    assert counts["unbacked_sentences"] == 0
    assert counts["has_unbacked_number"] is False
    assert counts["backed_sentence_rate"] == 1.0


def test_counts_unbacked_when_no_passage_supports_sentence():
    answer = "Gardening is a relaxing hobby enjoyed by many retirees worldwide."
    counts = sentence_attribution_counts(answer, [])
    assert counts["attributed_sentences"] == 0
    assert counts["unbacked_sentences"] == 1
    assert counts["backed_sentence_rate"] == 0.0


def test_counts_flags_unbacked_number():
    answer = "The temperature dropped to -268.93 degrees during the experiment."
    passage = _passage("S1", "p1", "Experiments require careful temperature control in the lab.")
    counts = sentence_attribution_counts(answer, [passage])
    assert counts["has_unbacked_number"] is True


def test_counts_denominator_zero_when_answer_is_only_short_non_claims():
    counts = sentence_attribution_counts("Great question!", [])
    assert counts["attributed_sentences"] == 0
    assert counts["unbacked_sentences"] == 0
    assert counts["backed_sentence_rate"] is None


def test_micro_average_weights_by_sentence_count_not_by_turn():
    # turn A: 1 attributed / 1 unbacked (rate 0.5); turn B: 9 attributed / 1
    # unbacked (rate 0.9). A plain average of rates would be 0.7; micro
    # averaging over pooled counts gives 10/12.
    counts = [
        {"attributed_sentences": 1, "unbacked_sentences": 1, "has_unbacked_number": False},
        {"attributed_sentences": 9, "unbacked_sentences": 1, "has_unbacked_number": False},
    ]
    assert micro_average_backed_sentence_rate(counts) == 10 / 12


def test_micro_average_none_when_every_denominator_zero():
    counts = [
        {"attributed_sentences": 0, "unbacked_sentences": 0, "has_unbacked_number": False},
    ]
    assert micro_average_backed_sentence_rate(counts) is None


def test_unbacked_number_rate_counts_turns_not_spans():
    counts = [
        {"attributed_sentences": 0, "unbacked_sentences": 2, "has_unbacked_number": True},
        {"attributed_sentences": 1, "unbacked_sentences": 0, "has_unbacked_number": False},
    ]
    assert unbacked_number_rate(counts) == 0.5


def test_unbacked_number_rate_empty_is_none():
    assert unbacked_number_rate([]) is None


def test_counts_from_event_all_backed():
    event = {
        "attributions": [
            {"sentence_span": [0, 10], "passage_id": "p1", "label": "S1", "score": 1.0,
             "model_cited": True},
        ],
        "unbacked": [],
    }
    counts = sentence_attribution_counts_from_event(event)
    assert counts["attributed_sentences"] == 1
    assert counts["unbacked_sentences"] == 0
    assert counts["has_unbacked_number"] is False
    assert counts["backed_sentence_rate"] == 1.0


def test_counts_from_event_flags_unbacked_number():
    event = {
        "attributions": [],
        "unbacked": [{"span": [0, 5], "reason": "unbacked_number"}],
    }
    counts = sentence_attribution_counts_from_event(event)
    assert counts["unbacked_sentences"] == 1
    assert counts["has_unbacked_number"] is True
    assert counts["backed_sentence_rate"] == 0.0


def test_counts_from_event_empty_denominator_is_none():
    counts = sentence_attribution_counts_from_event({"attributions": [], "unbacked": []})
    assert counts["backed_sentence_rate"] is None

"""Tests for tutor.retrieval.assessment.assess_evidence (pure, no I/O)."""

from __future__ import annotations

from types import SimpleNamespace

from tutor.retrieval.assessment import assess_evidence


def _passage(title: str, text: str):
    return SimpleNamespace(title=title, text=text)


class _Result:
    def __init__(self, passages):
        self.passages = passages


def test_empty_when_no_passages():
    result = _Result([])
    assessment = assess_evidence("What is the capital of France?", result)
    assert assessment.level == "empty"


def test_strong_when_top_passage_covers_key_terms():
    result = _Result(
        [
            _passage(
                "Square foot gardening",
                "Square foot gardening is a way to plan a small vegetable garden.",
            )
        ]
    )
    assessment = assess_evidence("how to squarefoot garden the right way?", result)
    # "squarefoot" itself won't match (it's fused), but "garden"/"gardening"
    # should stem-match via singularize, giving decent coverage once the
    # fused-word variant search (research.py) has actually found the
    # right article's title/text.
    assert assessment.level in ("strong", "weak")
    assert "garden" in {t for t in assessment.key_terms}


def test_weak_when_passages_are_off_topic():
    result = _Result(
        [_passage("Unrelated topic", "This article is about something else entirely.")]
    )
    assessment = assess_evidence("how does photosynthesis work in plants?", result)
    assert assessment.level == "weak"
    assert assessment.coverage < 0.5


def test_stem_match_garden_gardening():
    result = _Result([_passage("Gardening", "Gardening is the practice of growing plants.")])
    assessment = assess_evidence("what is square foot gardening?", result)
    assert "garden" in assessment.covered_terms or "gardening" in assessment.covered_terms
    assert assessment.coverage > 0.0


def test_question_fillers_excluded_from_key_terms():
    assessment = assess_evidence("how to squarefoot garden the right way?", _Result([]))
    assert "right" not in assessment.key_terms
    assert "way" not in assessment.key_terms


def test_no_key_terms_treated_as_strong_when_passages_present():
    result = _Result([_passage("Something", "Some text.")])
    assessment = assess_evidence("what about it?", result)
    assert assessment.level == "strong"


def test_only_checks_top_passages():
    # 4th passage covers the term, but only top 3 are checked.
    passages = [
        _passage("Irrelevant one", "nothing"),
        _passage("Irrelevant two", "nothing"),
        _passage("Irrelevant three", "nothing"),
        _passage("Volcano", "A volcano is an opening in the Earth's crust."),
    ]
    result = _Result(passages)
    assessment = assess_evidence("how do volcano eruptions happen?", result)
    assert assessment.level == "weak"


def test_to_dict_roundtrip_keys():
    result = _Result([_passage("Volcano", "A volcano erupts lava.")])
    assessment = assess_evidence("how do volcanoes erupt?", result)
    d = assessment.to_dict()
    assert set(d) == {"level", "reasons", "key_terms", "covered_terms", "coverage"}


def test_rewritten_queries_terms_replace_fused_original_terms():
    """The reassessment call after a forced rewrite must judge coverage
    against the REWRITTEN queries' own terms, not the original fused/
    misspelt "squarefoot" -- see docs/rewrite_on_weak_evidence.md."""
    result = _Result(
        [
            _passage(
                "Square foot gardening",
                "Square foot gardening is a method for planning small vegetable "
                "gardens using a grid.",
            )
        ]
    )
    rewritten_queries = [
        "how to square foot garden the right way",
        "square foot gardening basics",
        "square foot gardening guide step by step",
    ]
    rewritten = assess_evidence(
        "how to squarefoot garden the right way?",
        result,
        rewritten_queries=rewritten_queries,
    )
    # The fused original term pins coverage low or ambiguous; the rewritten
    # queries' own (correctly spelt) terms should give a clearly strong
    # verdict on the same evidence.
    assert rewritten.level == "strong"
    assert "squarefoot" not in rewritten.key_terms


def test_rewritten_queries_with_healthy_original_terms_kept():
    """A correctly-spelt original-question term with a healthy match is
    kept alongside the rewritten queries' terms, not dropped."""
    result = _Result(
        [
            _passage(
                "Square foot gardening",
                "Square foot gardening basics for a raised bed vegetable garden.",
            )
        ]
    )
    assessment = assess_evidence(
        "square foot garden basics",
        result,
        rewritten_queries=["square foot gardening basics"],
        healthy_terms={"garden": True, "basics": True},
    )
    assert assessment.level == "strong"


def test_no_rewritten_queries_is_backward_compatible():
    """Omitting the new parameters reproduces the exact old signature's
    behaviour."""
    result = _Result([_passage("Volcano", "A volcano is an opening in the crust.")])
    old = assess_evidence("What is a volcano?", result)
    new = assess_evidence("What is a volcano?", result, rewritten_queries=None)
    assert old.level == new.level
    assert old.key_terms == new.key_terms

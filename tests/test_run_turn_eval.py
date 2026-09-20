"""Unit tests (fakes only, no network) for the pure scoring/report parts
of ``eval/run_turn_eval.py``: ``score_answer``, ``aggregate``,
``render_report``, and ``load_questions``'s fixture-article filter."""

from __future__ import annotations

from eval.run_turn_eval import (
    AnswerRecord,
    aggregate,
    load_questions,
    render_report,
    score_answer,
)


def _record(**overrides) -> AnswerRecord:
    base = dict(
        question_id="q01",
        question="What is the Pythagorean theorem?",
        expected_paths=["pythagorean_theorem"],
        answer_text="",
        citations=[],
        evidence_dump=False,
    )
    base.update(overrides)
    return AnswerRecord(**base)


def test_score_answer_detects_a_cited_and_resolved_answer():
    record = _record(
        answer_text="A right triangle satisfies a^2+b^2=c^2 [S1].",
        citations=[{"label": "S1", "path": "pythagorean_theorem", "unresolved": False}],
    )
    scored = score_answer(record)
    assert scored["has_citation"] is True
    assert scored["n_labels"] == 1
    assert scored["all_resolve"] is True
    assert scored["on_topic_citation"] is True


def test_score_answer_flags_uncited_answer():
    record = _record(answer_text="A right triangle satisfies a^2+b^2=c^2.")
    scored = score_answer(record)
    assert scored["has_citation"] is False
    assert scored["all_resolve"] is False
    assert scored["on_topic_citation"] is False


def test_score_answer_flags_unresolved_label():
    record = _record(
        answer_text="See [S9].",
        citations=[{"label": "S9", "path": None, "unresolved": True}],
    )
    scored = score_answer(record)
    assert scored["has_citation"] is True
    assert scored["all_resolve"] is False
    assert scored["n_unresolved"] == 1


def test_score_answer_off_topic_citation_not_on_topic():
    record = _record(
        answer_text="See [S1].",
        citations=[{"label": "S1", "path": "unrelated_article", "unresolved": False}],
    )
    scored = score_answer(record)
    assert scored["on_topic_citation"] is False


def test_score_answer_detects_teaching_question():
    record = _record(answer_text="Can you tell me what a and b are here?")
    scored = score_answer(record)
    assert scored["taught"] is True


# ---------------------------------------------------------------------------
# 2026-09-20 evidence-dump follow-up: supported_citation_rate,
# evidence_dump_rate (docs/citation_experiment.md). Re-measurement with the
# live model is PENDING; these are pure functions exercised with fakes.
# ---------------------------------------------------------------------------


def test_score_answer_all_supported_true_when_every_citation_supported():
    record = _record(
        answer_text="Water boils at 100C [S1].",
        citations=[
            {"label": "S1", "path": "pythagorean_theorem", "unresolved": False, "supported": True}
        ],
    )
    scored = score_answer(record)
    assert scored["all_supported"] is True


def test_score_answer_all_supported_false_when_a_citation_is_unsupported():
    record = _record(
        answer_text="Water boils at 100C [S1].",
        citations=[
            {"label": "S1", "path": "pythagorean_theorem", "unresolved": False, "supported": False}
        ],
    )
    scored = score_answer(record)
    assert scored["all_supported"] is False


def test_score_answer_all_supported_false_when_uncited():
    record = _record(answer_text="Water boils at 100C.")
    scored = score_answer(record)
    assert scored["all_supported"] is False


def test_score_answer_carries_evidence_dump_flag():
    record = _record(answer_text="x [S1]", evidence_dump=True)
    assert score_answer(record)["evidence_dump"] is True
    record2 = _record(answer_text="x [S1]", evidence_dump=False)
    assert score_answer(record2)["evidence_dump"] is False


def test_aggregate_computes_supported_citation_rate_and_evidence_dump_rate():
    scores = [
        score_answer(
            _record(
                answer_text="cited [S1]",
                citations=[
                    {
                        "label": "S1",
                        "path": "pythagorean_theorem",
                        "unresolved": False,
                        "supported": True,
                    }
                ],
                evidence_dump=False,
            )
        ),
        score_answer(
            _record(
                answer_text="dumped [S1][S2][S3][S4]",
                citations=[
                    {
                        "label": f"S{i}",
                        "path": "pythagorean_theorem",
                        "unresolved": False,
                        "supported": False,
                    }
                    for i in range(1, 5)
                ],
                evidence_dump=True,
            )
        ),
    ]
    summary = aggregate(scores)
    assert summary["supported_citation_rate"] == 0.5
    assert summary["evidence_dump_rate"] == 0.5


def test_aggregate_empty_input_zeroes_new_rates_too():
    summary = aggregate([])
    assert summary["supported_citation_rate"] == 0.0
    assert summary["evidence_dump_rate"] == 0.0


def test_aggregate_computes_rates_over_multiple_scores():
    scores = [
        score_answer(_record(answer_text="cited [S1]", citations=[
            {"label": "S1", "path": "pythagorean_theorem", "unresolved": False}
        ])),
        score_answer(_record(answer_text="no citation here")),
    ]
    summary = aggregate(scores)
    assert summary["n"] == 2
    assert summary["citation_rate"] == 0.5
    assert summary["on_topic_rate"] == 0.5


def test_aggregate_empty_input_is_zeroed_not_a_crash():
    summary = aggregate([])
    assert summary["n"] == 0
    assert summary["citation_rate"] == 0.0


def test_render_report_produces_one_row_per_variant_in_order():
    summaries = {
        "baseline": aggregate([score_answer(_record(answer_text="x"))]),
        "variant_a": aggregate([score_answer(_record(answer_text="x [S1]", citations=[
            {"label": "S1", "path": "pythagorean_theorem", "unresolved": False}
        ]))]),
    }
    report = render_report(summaries)
    lines = report.splitlines()
    assert lines[0].startswith("| Variant |")
    assert lines[2].startswith("| baseline |")
    assert lines[3].startswith("| variant_a |")
    assert "1.00" in lines[3]


def test_render_report_includes_supported_citation_and_evidence_dump_rate_columns():
    summaries = {
        "baseline": aggregate(
            [
                score_answer(
                    _record(
                        answer_text="x [S1]",
                        citations=[
                            {
                                "label": "S1",
                                "path": "pythagorean_theorem",
                                "unresolved": False,
                                "supported": True,
                            }
                        ],
                        evidence_dump=False,
                    )
                )
            ]
        ),
    }
    report = render_report(summaries)
    assert "supported_citation_rate" in report
    assert "evidence_dump_rate" in report
    assert "1.00" in report.splitlines()[2]
    assert "0.00" in report.splitlines()[2]


def test_load_questions_filters_to_fixture_zim_articles():
    questions = load_questions()
    ids = {q["id"] for q in questions}
    # q01-q05's expected_paths exist in tests/zim_fixtures.py's fixture
    # ZIM; q06-q12 (photosynthesis, water cycle, WWII, ...) do not.
    assert ids == {"q01", "q02", "q03", "q04", "q05"}

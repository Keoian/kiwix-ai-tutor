"""Unit tests (pure parts only, no network) for eval/run_lesson_soak.py.

The live ``run_soak`` function drives a real llama-server and is exercised
manually (see docs/M5_report.md), not here: these tests cover script
building/cycling, calc scoring, aggregation, and report rendering with
fakes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_lesson_soak import (
    ScriptItem,
    TurnRecord,
    aggregate,
    build_script,
    cycle_script,
    render_report,
    score_calc,
    turns_dump_path,
    write_turns_dump,
)

# ---------------------------------------------------------------------------
# script
# ---------------------------------------------------------------------------


def test_build_script_has_two_subjects_and_a_realistic_mix():
    script = build_script()
    assert len(script) >= 30
    subjects = {item.subject for item in script}
    assert subjects == {"photosynthesis", "fractions"}

    kinds = {item.kind for item in script}
    assert kinds == {"text", "action"}

    actions = {item.action for item in script if item.kind == "action"}
    assert actions == {"simpler", "deeper", "hint"}

    calc_items = [item for item in script if item.expected_calc is not None]
    assert len(calc_items) >= 3


def test_cycle_script_repeats_when_n_exceeds_length():
    script = [ScriptItem("s", "text", text="a"), ScriptItem("s", "text", text="b")]
    out = cycle_script(script, 5)
    assert [i.text for i in out] == ["a", "b", "a", "b", "a"]


def test_cycle_script_zero_is_empty():
    script = [ScriptItem("s", "text", text="a")]
    assert cycle_script(script, 0) == []


def test_cycle_script_empty_script_raises_if_n_positive():
    with pytest.raises(ValueError):
        cycle_script([], 3)


# ---------------------------------------------------------------------------
# calc scoring
# ---------------------------------------------------------------------------


def test_score_calc_matches_number_in_prose():
    assert score_calc("3/4 as a percent is 75%.", 75.0) is True


def test_score_calc_matches_within_tolerance():
    assert score_calc("That comes out to about 0.63.", 0.625) is True


def test_score_calc_rejects_wrong_number():
    assert score_calc("The answer is 42.", 75.0) is False


def test_score_calc_no_numbers_is_false():
    assert score_calc("I'm not sure.", 75.0) is False


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def _rec(**kwargs) -> TurnRecord:
    base = dict(
        index=1,
        subject="photosynthesis",
        input_kind="text",
        input_text="x",
        wall_s=1.0,
        ttft_s=0.5,
        tt_tool_s=None,
        route="preretrieve",
        research_statuses=["ok"],
        research_elapsed_s=[0.2],
        citations=1,
        citations_resolved=1,
        uncited=False,
        calc_calls=0,
        tokens_used=1000,
        cached_tokens=800,
        eviction_events=0,
        status="ok",
        answer_text="Plants use sunlight [S1].",
        labels=["S1"],
        unsupported_labels=[],
        citation_quality="ok",
        evidence_dump=False,
        passages=[],
    )
    base.update(kwargs)
    return TurnRecord(**base)


def test_aggregate_counts_ok_and_errored():
    records = [_rec(index=1, status="ok"), _rec(index=2, status="error", error="boom")]
    summary = aggregate(records)
    assert summary["turns_completed"] == 2
    assert summary["turns_ok"] == 1
    assert summary["turns_errored"] == 1
    assert summary["errors"] == [{"index": 2, "status": "error", "error": "boom"}]


def test_aggregate_latency_percentiles():
    records = [_rec(index=i, wall_s=float(i), ttft_s=float(i) / 2) for i in range(1, 11)]
    summary = aggregate(records)
    assert summary["latency"]["total_p50"] == pytest.approx(5.5, abs=0.6)
    assert summary["latency"]["ttft_p95"] is not None


def test_aggregate_citation_and_uncited_rates_only_over_factual_turns():
    records = [
        _rec(index=1, route="preretrieve", citations=1, citations_resolved=1, uncited=False),
        _rec(index=2, route="preretrieve", citations=0, citations_resolved=0, uncited=True),
        _rec(index=3, route="action", citations=0, citations_resolved=0, uncited=False),
    ]
    summary = aggregate(records)
    assert summary["factual_turns"] == 2
    assert summary["citation_rate"] == pytest.approx(0.5)
    assert summary["uncited_rate"] == pytest.approx(0.5)


def test_aggregate_calc_accuracy():
    records = [
        _rec(index=1, expected_calc=75.0, calc_correct=True),
        _rec(index=2, expected_calc=80.0, calc_correct=False),
        _rec(index=3, expected_calc=None, calc_correct=None),
    ]
    summary = aggregate(records)
    assert summary["calc_items"] == 2
    assert summary["calc_correct"] == 1
    assert summary["calc_accuracy"] == pytest.approx(0.5)


def test_aggregate_research_status_mix_and_mean_elapsed():
    records = [
        _rec(index=1, research_statuses=["ok"], research_elapsed_s=[0.1]),
        _rec(index=2, research_statuses=["partial", "ok"], research_elapsed_s=[0.3, 0.2]),
    ]
    summary = aggregate(records)
    assert summary["research_status_mix"] == {"ok": 2, "partial": 1}
    assert summary["research_mean_elapsed_s"] == pytest.approx(0.2, abs=0.001)


def test_aggregate_slow_turns_flagged_over_five_minutes():
    records = [_rec(index=1, wall_s=301.0), _rec(index=2, wall_s=10.0)]
    summary = aggregate(records)
    assert summary["slow_turns_over_5min"] == [1]


def test_aggregate_cache_hit_ratio_mean():
    records = [
        _rec(index=1, tokens_used=1000, cached_tokens=900),
        _rec(index=2, tokens_used=2000, cached_tokens=1000),
    ]
    summary = aggregate(records)
    assert summary["cache_hit_ratio_mean"] == pytest.approx((0.9 + 0.5) / 2)


def test_turn_record_is_factual_reflects_route():
    assert _rec(route="preretrieve").is_factual is True
    assert _rec(route="preretrieve_deep").is_factual is True
    assert _rec(route="action").is_factual is False
    assert _rec(route=None).is_factual is False


def test_aggregate_cited_and_supported_rates_over_factual_turns_only():
    records = [
        _rec(index=1, route="preretrieve", labels=["S1"], unsupported_labels=[]),
        _rec(index=2, route="preretrieve", labels=["S1"], unsupported_labels=["S1"]),
        _rec(index=3, route="preretrieve", labels=[], unsupported_labels=[], uncited=True),
        _rec(index=4, route="action", labels=[], unsupported_labels=[]),
    ]
    summary = aggregate(records)
    assert summary["factual_turns"] == 3
    assert summary["cited_turns"] == 2
    assert summary["cited_rate"] == pytest.approx(2 / 3)
    assert summary["supported_turns"] == 1
    assert summary["supported_rate"] == pytest.approx(1 / 3)
    assert summary["uncited_rate"] == pytest.approx(1 / 3)
    # documented relationship: cited_rate == 1 - uncited_rate
    assert summary["cited_rate"] == pytest.approx(1 - summary["uncited_rate"])


def test_aggregate_evidence_dump_turns_counted_over_factual_only():
    records = [
        _rec(index=1, route="preretrieve", evidence_dump=True),
        _rec(index=2, route="preretrieve", evidence_dump=False),
        _rec(index=3, route="action", evidence_dump=True),
    ]
    summary = aggregate(records)
    assert summary["evidence_dump_turns"] == 1


def test_aggregate_empty_records_does_not_crash():
    summary = aggregate([])
    assert summary["turns_completed"] == 0
    assert summary["latency"]["ttft_p50"] is None
    assert summary["cache_hit_ratio_mean"] is None
    assert summary["calc_accuracy"] is None


# ---------------------------------------------------------------------------
# backed_sentence_rate / unbacked_number_rate (docs/attribution_design.md,
# eval.attribution_scoring), over factual turns only, computed from
# answer_text + passages via tutor.app.citations.attribute_sentences.
# ---------------------------------------------------------------------------

_PASSAGE = {
    "label": "S1",
    "id": "p1",
    "title": "Photosynthesis",
    "path": "Bio/Photosynthesis",
    "text": "Plants use sunlight to make food through photosynthesis.",
}


def test_aggregate_backed_sentence_rate_over_factual_turns():
    records = [
        _rec(
            index=1,
            route="preretrieve",
            answer_text="Plants use sunlight to make food [S1].",
            labels=["S1"],
            passages=[_PASSAGE],
        ),
        _rec(
            index=2,
            route="preretrieve",
            answer_text="It happened in exactly the year 1523.",
            labels=[],
            uncited=True,
            passages=[_PASSAGE],
        ),
        # non-factual turn: excluded from the denominator regardless of text.
        _rec(index=3, route="action", answer_text="It happened in the year 1523.", passages=[]),
    ]
    summary = aggregate(records)
    assert summary["backed_sentence_rate"] == pytest.approx(0.5)
    assert summary["unbacked_number_rate"] == pytest.approx(0.5)


def test_aggregate_backed_sentence_rate_none_when_no_factual_turns():
    records = [_rec(index=1, route="action")]
    summary = aggregate(records)
    assert summary["backed_sentence_rate"] is None
    assert summary["unbacked_number_rate"] is None


def test_write_turns_dump_carries_passages_for_offline_rescoring(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [_rec(index=1, passages=[_PASSAGE])]
    out_path = str(tmp_path / "docs" / "fake_soak.md")
    dump_path = write_turns_dump(records, out_path)
    payload = json.loads(Path(dump_path).read_text(encoding="utf-8"))
    assert payload[0]["passages"] == [_PASSAGE]


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------


def test_render_report_includes_gate_checklist_and_measured_numbers():
    records = [_rec(index=i) for i in range(1, 11)]
    summary = aggregate(records)
    resource_samples = [
        {
            "t": 0, "rss_mb": 200.0, "thread_count": 10, "open_files": 5,
            "child_process_count": 1, "llm_healthy": True,
        },
        {
            "t": 30, "rss_mb": 210.0, "thread_count": 11, "open_files": 6,
            "child_process_count": 1, "llm_healthy": True,
        },
    ]
    meta = {
        "generated_at": "2026-09-20T00:00:00",
        "config": "config/dev.toml",
        "minutes_requested": 30,
        "stopped_early": False,
        "resume_check": "PASS (measured): resumed cleanly.",
        "gate_status": "PASS (measured, no turn >5min, soak ran to completion)",
    }
    report = render_report(summary, resource_samples, meta)

    assert "M5 gate checklist" in report
    assert "turns completed: **10**" in report
    assert "RSS MB: start=200.000" in report
    assert "PASS (measured): resumed cleanly." in report
    assert "Deferred to the Dell (M6)" in report


def test_render_report_handles_no_resource_samples():
    summary = aggregate([])
    report = render_report(summary, [], {"generated_at": "t", "resume_check": "SKIPPED"})
    assert "no resource samples were collected" in report


def test_render_report_includes_citations_table_for_factual_turns():
    records = [
        _rec(index=1, route="preretrieve", labels=["S1"], unsupported_labels=[]),
        _rec(index=2, route="preretrieve", labels=[], unsupported_labels=[], uncited=True),
    ]
    summary = aggregate(records)
    report = render_report(summary, [], {"generated_at": "t", "resume_check": "SKIPPED"})
    assert "Citations (factual turns only)" in report
    assert "cited_turns / cited_rate" in report
    assert "supported_turns / supported_rate" in report
    assert "evidence_dump_turns" in report
    assert "1 - cited_rate" in report or "1 - uncited_rate" in report


def test_render_report_includes_backed_sentence_and_unbacked_number_rates():
    records = [
        _rec(
            index=1,
            route="preretrieve",
            answer_text="Plants use sunlight to make food [S1].",
            labels=["S1"],
            passages=[_PASSAGE],
        ),
    ]
    summary = aggregate(records)
    report = render_report(summary, [], {"generated_at": "t", "resume_check": "SKIPPED"})
    assert "backed_sentence_rate" in report
    assert "unbacked_number_rate" in report


# ---------------------------------------------------------------------------
# per-turn JSON dump (for later re-scoring)
# ---------------------------------------------------------------------------


def test_turns_dump_path_mirrors_stem_under_data_dir():
    assert turns_dump_path("docs/x/q1_soak10.md") == Path("data/q1_soak10.turns.json")


def test_write_turns_dump_writes_json_with_answer_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [
        _rec(index=1, answer_text="Plants make sugar [S1]."),
        _rec(index=2, answer_text="Because sunlight."),
    ]
    out_path = str(tmp_path / "docs" / "q1_soak10.md")
    dump_path = write_turns_dump(records, out_path)
    assert dump_path == Path("data/q1_soak10.turns.json")
    assert dump_path.exists()
    loaded = json.loads(dump_path.read_text(encoding="utf-8"))
    assert len(loaded) == 2
    assert loaded[0]["answer_text"] == "Plants make sugar [S1]."
    assert loaded[0]["is_factual"] is True
    assert loaded[1]["answer_text"] == "Because sunlight."


# ---------------------------------------------------------------------------
# regression: the per-turn recorder-call drain pattern used in run_soak
# must not alias the buffer it clears (found live 2026-09-20: the M5 soak's
# research_status_mix/research_mean_elapsed_s came back empty because
# ``calls_for_turn, recorder_calls[:] = recorder_calls, []`` bound
# ``calls_for_turn`` to the same list object that the in-place clear then
# wiped). The fix drains via ``list(...)`` + ``.clear()``.
# ---------------------------------------------------------------------------


def test_drain_pattern_copies_before_clearing():
    recorder_calls = [("ok", 0.1), ("partial", 0.2)]

    calls_for_turn = list(recorder_calls)
    recorder_calls.clear()

    assert calls_for_turn == [("ok", 0.1), ("partial", 0.2)]
    assert recorder_calls == []


def test_broken_drain_pattern_would_have_aliased_and_lost_data():
    """Documents the exact bug that shipped in the first soak run: this
    is the OLD (buggy) pattern, kept here only to pin down why it failed."""
    recorder_calls = [("ok", 0.1), ("partial", 0.2)]

    calls_for_turn, recorder_calls[:] = recorder_calls, []

    # The bug: clearing recorder_calls in place also empties calls_for_turn,
    # since both names refer to the same list object.
    assert calls_for_turn == []

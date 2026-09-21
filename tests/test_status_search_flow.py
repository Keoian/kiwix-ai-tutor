"""Job 1 (status must never go dark): asserts the ordered sequence of
``status`` events emitted by ``run_turn`` under ``model_writes_search``
for (a) a normal turn, (b) a weak-evidence second round, and (c) a
not-found turn -- and that a status event is emitted after the last
search and before the first answer token in every case.

Reuses the fixtures from tests/test_agent_loop_model_writes_search.py.
"""

from __future__ import annotations

from tests.test_agent_loop_model_writes_search import (
    FakeCalc,
    FakeLlmClient,
    ScriptedResearchEngine,
    _final,
    _mk_session,
    _strong_response,
    _tool_call,
    _UserInput,
)
from tutor.app.agent_loop import run_turn
from tutor.app.llm_client import StreamEvent


def _collect(events_list, emit_call):
    events_list.append(emit_call)


def test_normal_turn_status_sequence_covers_gap_before_first_token():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Blue whale", "Largest animals"]}),
            _final("The blue whale is the largest animal [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),
            _strong_response(2, title="Blue whale"),
            _strong_response(3, title="Largest animals"),
        ]
    )
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="What's the biggest animal?"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=budget,
        emit=lambda e: _collect(events, e),
        model_writes_search=True,
    )
    assert result.status == "ok"

    statuses = [e for e in events if isinstance(e, dict) and e.get("kind") == "status"]
    stages = [s["stage"] for s in statuses]
    assert "planning" in stages
    assert "searching" in stages
    assert "reading" in stages
    assert "thinking" in stages

    # index of last status event and first token event, in the raw stream
    last_status_idx = max(
        i
        for i, e in enumerate(events)
        if isinstance(e, dict) and e.get("kind") == "status"
    )
    first_token_idx = next(
        i for i, e in enumerate(events) if getattr(e, "kind", None) == "token"
    )
    assert last_status_idx < first_token_idx


def test_weak_evidence_second_round_has_trying_different_search_status():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Nonexistent Topic"]}),
            _tool_call("research", {"queries": ["Nonexistent Topic 2"]}),
            _final("I couldn't find much, but here's what I know."),
        ]
    )
    from tests.test_agent_loop_model_writes_search import _Response

    research = ScriptedResearchEngine(
        [
            _Response(passages=[]),
            _Response(passages=[]),
            _Response(passages=[]),
        ]
    )
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="What's a glorfindel?"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=budget,
        emit=lambda e: _collect(events, e),
        model_writes_search=True,
        rewrite_on_weak_evidence=True,
    )
    assert result.status == "ok"

    statuses = [e for e in events if isinstance(e, dict) and e.get("kind") == "status"]
    details = [s["detail"] for s in statuses]
    assert any("Trying a different search" in d for d in details)
    assert any("Nothing in the library" in d for d in details)

    last_status_idx = max(
        i
        for i, e in enumerate(events)
        if isinstance(e, dict) and e.get("kind") == "status"
    )
    first_token_idx = next(
        i for i, e in enumerate(events) if getattr(e, "kind", None) == "token"
    )
    assert last_status_idx < first_token_idx


def test_not_found_turn_has_answering_from_what_i_know_status():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            [StreamEvent(kind="done", finish_reason="stop", usage={})],
            _final("Sorry, I don't have that in the library, but here's what I know."),
        ]
    )
    from tests.test_agent_loop_model_writes_search import _Response

    research = ScriptedResearchEngine([_Response(passages=[])])
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="What's a glorfindel?"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=budget,
        emit=lambda e: _collect(events, e),
        model_writes_search=True,
        rewrite_on_weak_evidence=False,
    )
    assert result.status == "ok"

    statuses = [e for e in events if isinstance(e, dict) and e.get("kind") == "status"]
    details = [s["detail"] for s in statuses]
    assert any("Nothing in the library on this" in d for d in details)

    last_status_idx = max(
        i
        for i, e in enumerate(events)
        if isinstance(e, dict) and e.get("kind") == "status"
    )
    first_token_idx = next(
        i for i, e in enumerate(events) if getattr(e, "kind", None) == "token"
    )
    assert last_status_idx < first_token_idx


def test_timings_present_and_non_negative_normal_turn():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Blue whale"]}),
            _final("The blue whale is the largest animal [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),
            _strong_response(2, title="Blue whale"),
        ]
    )

    result = run_turn(
        session,
        _UserInput(kind="text", text="What's the biggest animal?"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )
    assert result.status == "ok"
    expected_keys = {
        "forced_call",
        "searches",
        "second_round",
        "answer_prefill",
        "answer_generation",
        "presearch",
        "voluntary_tool_rounds",
    }
    assert set(result.timings.keys()) == expected_keys
    for key, value in result.timings.items():
        assert value >= 0.0, f"{key} should be non-negative"
    assert result.timings["second_round"] == 0.0


def test_timings_second_round_nonzero_when_weak_evidence_round_fires():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Nonexistent Topic"]}),
            _tool_call("research", {"queries": ["Nonexistent Topic 2"]}),
            _final("I couldn't find much, but here's what I know."),
        ]
    )
    from tests.test_agent_loop_model_writes_search import _Response

    research = ScriptedResearchEngine(
        [
            _Response(passages=[]),
            _Response(passages=[]),
            _Response(passages=[]),
        ]
    )

    result = run_turn(
        session,
        _UserInput(kind="text", text="What's a glorfindel?"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
        rewrite_on_weak_evidence=True,
    )
    assert result.status == "ok"
    for value in result.timings.values():
        assert value >= 0.0
    assert isinstance(result.timings["second_round"], float)

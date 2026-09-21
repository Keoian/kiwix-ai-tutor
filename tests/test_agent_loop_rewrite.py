"""Tests for the forced query-rewrite-on-weak-evidence mechanism
(docs/rewrite_on_weak_evidence.md).

All fakes; no live LLM. Mirrors tests/test_agent_loop_promptlog.py's
pattern (a real ``Session``/``PromptLog``, a scripted fake LLM asserting
on the actual wire messages, fake research/calc).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tutor.app.agent_loop import _MERGE_CAP, _merge_dedupe_passages, run_turn
from tutor.app.llm_client import StreamEvent
from tutor.app.prompt import Budget, serialize_messages
from tutor.app.session import Session
from tutor.retrieval.assessment import assess_evidence


def _count_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


@dataclass
class _UserInput:
    kind: str
    text: str | None = None
    action: str | None = None


class FakeLlmClient:
    def __init__(self, scripts: list[list[StreamEvent]]):
        self._scripts = list(scripts)
        self.calls: list[list[dict]] = []
        self.tool_choice_calls: list[object] = []
        self.response_format_calls: list[object] = []
        self.max_tokens_calls: list[int | None] = []

    def stream_chat(
        self,
        messages,
        *,
        max_tokens=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        cancel=None,
        temperature=None,
    ):
        self.calls.append([dict(m) for m in messages])
        self.tool_choice_calls.append(tool_choice)
        self.response_format_calls.append(response_format)
        self.max_tokens_calls.append(max_tokens)
        script = self._scripts.pop(0)
        yield from script

    @property
    def call_count(self) -> int:
        return len(self.calls)


@dataclass
class _Passage:
    label: str
    passage_id: str
    title: str = "Title"
    path: str = "A/Title"
    text: str = "Some evidence text about the topic in detail. " * 10
    kind: str = "article"


@dataclass
class _Response:
    status: str = "ok"
    passages: list = field(default_factory=list)
    assessment: object = None
    corrected_terms: dict = field(default_factory=dict)


class ScriptedResearchEngine:
    """Returns one scripted ``_Response`` per call, in order, and computes
    a real ``assess_evidence`` result against ``question`` for each so the
    weak/strong assertions below reflect the real threshold, not a faked
    level."""

    def __init__(self, responses: list[_Response]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def research(self, query, *, topic_hint=None, keywords=None):
        self.calls.append({"query": query, "topic_hint": topic_hint, "keywords": keywords})
        response = self._responses.pop(0)
        response.assessment = assess_evidence(query, response)
        return response

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeCalc:
    def evaluate(self, expression: str, **kwargs):
        return {"ok": True, "result": "4"}


def _token(text: str) -> StreamEvent:
    return StreamEvent(kind="token", text=text)


def _done(finish_reason="stop") -> StreamEvent:
    return StreamEvent(kind="done", finish_reason=finish_reason, usage={})


def _final(text: str) -> list[StreamEvent]:
    return [_token(text), _done()]


def _tool_call(name: str, arguments: dict, call_id="call_1") -> list[StreamEvent]:
    return [
        StreamEvent(kind="tool_call", id=call_id, name=name, arguments_json=json.dumps(arguments)),
        _done(finish_reason="tool_calls"),
    ]


def _mk_session() -> tuple[Session, Budget]:
    session = Session(count_tokens=_count_tokens, subject_hint="astronomy")
    return session, Budget()


_EMPTY_RESPONSE = _Response(passages=[])


def _strong_response(n: int) -> _Response:
    return _Response(
        passages=[
            _Passage(
                label=f"S{n}",
                passage_id=f"pid-{n}",
                title="Solar system",
                text="The solar system has eight planets orbiting the sun. " * 10,
            )
        ]
    )


def test_weak_evidence_forces_research_tool_choice_and_rewrites():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["solar system planets"]}),
            _final("The solar system has eight planets [S1]."),
        ]
    )
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _strong_response(1)])
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="wut is teh solarsystem"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )

    assert result.status == "ok"
    # Forced call used tool_choice (primary path), not the grammar fallback.
    assert llm.tool_choice_calls[0] == {
        "type": "function",
        "function": {"name": "research"},
    }
    assert llm.response_format_calls[0] is None
    assert llm.max_tokens_calls[0] == 96
    # Host executed the rewritten query for real.
    assert research.calls[1]["query"] == "solar system planets"
    assert result.evidence["level_before"] == "empty"
    assert result.evidence["level_after"] == "strong"
    assert result.evidence["rewritten_queries"] == ["solar system planets"]

    rendered = session.log.render()
    roles = [m["role"] for m in rendered]
    # user (with host note) -> assistant tool_calls -> tool result -> assistant answer
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert "Host note" in rendered[1]["content"]
    assert roles.count("assistant") >= 1
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    assert any(
        "Searched for: solar system planets" in (m.get("content") or "")
        for m in tool_messages
    )


def test_still_weak_after_rewrite_gets_not_found_instruction():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["xyzzy topic"]}),
            _final("I could not find this in the library."),
        ]
    )
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _EMPTY_RESPONSE])
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="tell me about xyzzy"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )

    assert result.status == "ok"
    assert result.evidence["level_before"] == "empty"
    assert result.evidence["level_after"] == "empty"
    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "could not find this in the library" in joined
    assert "unchecked" in joined


def test_tool_choice_rejected_falls_back_to_json_schema():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            [StreamEvent(kind="error", error="HTTP 400")],
            [_token(json.dumps({"queries": ["solar system"]})), _done()],
            _final("Answer [S1]."),
        ]
    )
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _strong_response(1)])
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="wut is teh solarsystem"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )

    assert result.status == "ok"
    assert llm.tool_choice_calls[0] == {"type": "function", "function": {"name": "research"}}
    # Fallback call used response_format instead of tool_choice.
    assert llm.response_format_calls[1]["type"] == "json_schema"
    assert result.evidence["rewritten_queries"] == ["solar system"]


def test_strong_evidence_never_triggers_rewrite():
    session, budget = _mk_session()
    llm = FakeLlmClient([_final("The sun is a star [S1].")])
    research = ScriptedResearchEngine([_strong_response(1)])
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="what is the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )

    assert result.status == "ok"
    assert research.call_count == 1
    assert llm.call_count == 1
    assert result.evidence["level_before"] == "strong"
    assert result.evidence["level_after"] == "strong"
    assert result.evidence["rewritten_queries"] == []


def test_action_route_never_rewrites():
    session, budget = _mk_session()
    llm = FakeLlmClient([_final("Sure, here it is simpler.")])
    research = ScriptedResearchEngine([])
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="action", action="simpler"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )

    assert result.status == "ok"
    assert research.call_count == 0
    assert result.evidence is None


def test_setting_off_is_byte_identical_to_no_rewrite_support():
    """rewrite_on_weak_evidence=False must reproduce today's prompt bytes
    exactly for a weak-evidence turn (no host note, no forced call, plain
    evidence packet appended even though it's weak, same as before this
    feature existed)."""
    session, budget = _mk_session()
    llm = FakeLlmClient([_final("Best I can say [S1].")])
    research = ScriptedResearchEngine([_EMPTY_RESPONSE])
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="wut is teh solarsystem"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_weak_evidence=False,
    )

    assert result.status == "ok"
    assert research.call_count == 1
    assert llm.call_count == 1
    rendered = session.log.render()
    assert "Host note" not in rendered[1]["content"]
    assert rendered[1]["content"] == "wut is teh solarsystem"


def test_rewritten_evidence_is_evictable_and_prefix_stable():
    """The forced call's assistant tool_calls + tool result are ordinary
    log entries: they survive eviction like any other turn, and turn N+1's
    prompt starts with the exact bytes of turn N's final rendered prompt
    (the byte-prefix property), even though turn N included a rewrite
    round."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["solar system planets"]}),
            _final("The solar system has eight planets [S1]."),
            _final("Follow-up answer [S1]."),
        ]
    )
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _strong_response(1), _strong_response(2)])
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="wut is teh solarsystem"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )
    turn1_final_prompt = serialize_messages(llm.calls[-1] + [
        {"role": "assistant", "content": "The solar system has eight planets [S1]."}
    ])

    run_turn(
        session,
        _UserInput(kind="text", text="What else is in the solar system?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )
    turn2_first_prompt = serialize_messages(llm.calls[-1])

    assert turn2_first_prompt.startswith(turn1_final_prompt)


# ---------------------------------------------------------------------------
# RRF merge (Merge / re-assess rule fix) -- shape reproduces the live smoke
# bug: three rewritten queries each rank the target article, but never
# first, alongside disambiguation/generic-title noise.
# ---------------------------------------------------------------------------

def _p(pid, title, rank_tag):
    return {
        "id": pid,
        "label": pid,
        "title": title,
        "path": title,
        "text": f"Evidence about {title} ({rank_tag}). " * 5,
        "kind": "article",
    }


def test_rrf_merge_puts_multi_query_target_article_first():
    rewritten_queries = [
        "how to square foot garden the right way",
        "square foot gardening basics",
        "square foot gardening guide step by step",
    ]
    # Reproduces the smoke bug's shape: target last in every list.
    list1 = [
        _p("sixfoot-1", "6 Foot 7 Foot", "q1"),
        _p("garden-1", "Garden", "q1"),
        _p("sfg-1", "Square foot gardening", "q1"),
    ]
    list2 = [
        _p("square-2", "Square (disambiguation)", "q2"),
        _p("garden-2", "Garden", "q2"),
        _p("sfg-2", "Square foot gardening", "q2"),
    ]
    list3 = [
        _p("sixfoot-3", "6 Foot 7 Foot", "q3"),
        _p("square-3", "Square (disambiguation)", "q3"),
        _p("sfg-3", "Square foot gardening", "q3"),
    ]

    merged = _merge_dedupe_passages(
        [list1, list2, list3], _MERGE_CAP, rewritten_queries=rewritten_queries
    )

    assert merged[0]["title"] == "Square foot gardening"


def test_rrf_merge_multi_query_hit_outranks_single_query_hit():
    list1 = [_p("a1", "Some Other Article", "q1")]  # rank 0 in one query only
    list2 = [
        _p("b1", "Random Noise Article", "q2"),
        _p("sfg", "Square foot gardening", "q2"),
    ]
    list3 = [
        _p("c1", "Another Article", "q3"),
        _p("sfg", "Square foot gardening", "q3"),
    ]
    merged = _merge_dedupe_passages([list1, list2, list3], _MERGE_CAP)
    titles = [p["title"] for p in merged]
    assert titles.index("Square foot gardening") < titles.index("Some Other Article")


def test_rrf_merge_cap_unchanged():
    lists = [[_p(f"id{i}", f"Title {i}", "q") for i in range(12)]]
    merged = _merge_dedupe_passages(lists, _MERGE_CAP)
    assert len(merged) == _MERGE_CAP


def test_rrf_merge_deterministic_tie_break():
    list1 = [_p("z1", "Zebra", "q"), _p("a1", "Aardvark", "q")]
    list2 = [_p("z1", "Zebra", "q"), _p("a1", "Aardvark", "q")]
    merged1 = _merge_dedupe_passages([list1, list2], _MERGE_CAP)
    merged2 = _merge_dedupe_passages([list2, list1], _MERGE_CAP)
    assert [p["id"] for p in merged1] == [p["id"] for p in merged2]

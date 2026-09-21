"""Tests for the unconditional follow-up-turn query-rewrite mechanism
(docs/rewrite_on_weak_evidence.md, "Follow-up rewrite"): on every turn
after the lesson's first, the host forces a model-written query-rewrite
round before trusting the raw pre-search, regardless of how strong that
raw pre-search looks -- never gated by a word-list/pronoun detector.

All fakes; no live LLM. Mirrors tests/test_agent_loop_rewrite.py's
pattern (a real ``Session``/``PromptLog``, a scripted fake LLM asserting
on the actual wire messages, fake research/calc).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tutor.app.agent_loop import run_turn
from tutor.app.llm_client import StreamEvent
from tutor.app.prompt import Budget
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
        script = self._scripts.pop(0)
        yield from script

    @property
    def call_count(self) -> int:
        return len(self.calls)


@dataclass
class _Passage:
    label: str
    passage_id: str
    title: str = "Solar system"
    path: str = "A/Solar_system"
    text: str = "The solar system has eight planets orbiting the sun. " * 10
    kind: str = "article"


@dataclass
class _Response:
    status: str = "ok"
    passages: list = field(default_factory=list)
    assessment: object = None
    corrected_terms: dict = field(default_factory=dict)


class ScriptedResearchEngine:
    """One scripted ``_Response`` per ``research()`` call, in order (also
    used, in order, for each query in a ``research_many`` batch); real
    ``assess_evidence`` computed per call against the actual query text."""

    def __init__(self, responses: list[_Response]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def research(self, query, *, topic_hint=None, keywords=None):
        self.calls.append({"query": query, "topic_hint": topic_hint, "keywords": keywords})
        response = self._responses.pop(0)
        response.assessment = assess_evidence(query, response)
        return response

    def research_many(self, queries, *, topic_hint=None):
        return [
            {
                "query": q,
                "status": "ok",
                "response": self.research(q, topic_hint=topic_hint),
                "error": None,
            }
            for q in queries
        ]

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


def _strong_response(n: int) -> _Response:
    return _Response(passages=[_Passage(label=f"S{n}", passage_id=f"pid-{n}")])


def test_followup_fires_on_second_turn_even_with_strong_raw_evidence():
    """Turn 1: strong raw evidence, no rewrite. Turn 2: the raw pre-search
    ALSO comes back strong, but a follow-up turn must never trust it --
    the host forces a rewrite round first regardless, and the rewrite's
    own evidence (a distinct passage id) leads the final evidence packet,
    not the raw pre-search's own passage."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets in the solar system"]}),
            _final("There are also moons and asteroids [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1),  # turn 1 raw pre-search
            _strong_response(2),  # turn 2 raw pre-search (never trusted directly)
            _strong_response(3),  # turn 2 rewrite's own query
        ]
    )
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="what is the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )

    result = run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
    )

    assert result.status == "ok"
    # The forced round used tool_choice (turn 2's first LLM call).
    assert llm.tool_choice_calls[1] == {"type": "function", "function": {"name": "research"}}
    assert result.evidence["rewritten_queries"] == ["moons and planets in the solar system"]
    rendered = session.log.render()
    # rendered[0]=system, [1]=user(turn1), [2]=tool(turn1 evidence),
    # [3]=assistant(turn1 answer), [4]=user(turn2, with host note).
    assert "Host note" in rendered[4]["content"]
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "Searched for: moons and planets in the solar system" in joined
    # The rewrite's own evidence ([S3]) leads; the raw turn-2 pre-search's
    # passage ([S2]) is only ever used as backfill behind it, never ahead.
    assert "[S3]" in joined
    assert joined.index("[S3]") < joined.index("[S2]")
    assert "direct answer" in joined.lower() or "directly first" in joined.lower()


def test_followup_never_fires_on_first_turn():
    session, budget = _mk_session()
    llm = FakeLlmClient([_final("The solar system has eight planets [S1].")])
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
    assert llm.call_count == 1
    assert result.evidence["rewritten_queries"] == []
    rendered = session.log.render()
    assert "Host note" not in rendered[1]["content"]


def test_setting_off_disables_followup_rewrite_on_second_turn():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _final("There are also moons and asteroids [S1]."),
        ]
    )
    research = ScriptedResearchEngine([_strong_response(1), _strong_response(2)])
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="what is the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_followup=False,
    )

    result = run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        rewrite_on_followup=False,
    )

    assert result.status == "ok"
    assert llm.call_count == 2  # no extra forced-rewrite round
    rendered = session.log.render()
    assert "Host note" not in rendered[4]["content"]

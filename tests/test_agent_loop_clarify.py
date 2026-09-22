"""Tests for the host-driven "Did you mean X?" clarify wiring in
tutor.app.agent_loop.run_turn (owner request 2026-09-21; see
tutor.app.clarify for the pure parsing/classification helpers, tested
separately in tests/test_clarify.py).

Pure unit tests: a scripted FAKE llm client and a FAKE research engine
(no network, no real model, no real ZIM archive), following the same
fake patterns as tests/test_agent_loop.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tutor.app.agent_loop import TurnResult, run_turn
from tutor.app.llm_client import StreamEvent
from tutor.app.prompt import Budget
from tutor.app.session import Session

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLlmClient:
    def __init__(self, scripts: list[list[StreamEvent]]):
        self._scripts = list(scripts)
        self.calls: list[list[dict]] = []

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
        self.calls.append(list(messages))
        if not self._scripts:
            raise AssertionError("FakeLlmClient: no more scripted responses")
        script = self._scripts.pop(0)
        yield from script

    @property
    def call_count(self) -> int:
        return len(self.calls)


@dataclass
class _Assessment:
    level: str
    covered_terms: frozenset = frozenset()


@dataclass
class _Passage:
    label: str
    passage_id: str = "p1"
    title: str = "Title"
    path: str = "A/Title"
    text: str = "Some evidence text."
    kind: str = "article"


@dataclass
class _Response:
    status: str = "ok"
    passages: list = field(default_factory=list)
    assessment: Any = None
    unknown_terms: list = field(default_factory=list)
    corrected_terms: dict = field(default_factory=dict)


class FakeResearchEngine:
    def __init__(
        self, by_query: dict[str, _Response] | None = None, default: _Response | None = None
    ):
        self.by_query = by_query or {}
        self.default = default or _Response(passages=[], assessment=_Assessment("empty"))
        self.calls: list[str] = []

    def research(self, query: str, *, topic_hint: str | None = None, keywords=None):
        self.calls.append(query)
        return self.by_query.get(query, self.default)

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeCalc:
    def evaluate(self, expression: str, **kwargs):
        return {"ok": True, "result": "0"}


def _token(text: str) -> StreamEvent:
    return StreamEvent(kind="token", text=text)


def _done(finish_reason="stop") -> StreamEvent:
    return StreamEvent(kind="done", finish_reason=finish_reason, usage={})


def _reply_script(text: str) -> list[StreamEvent]:
    return [_token(text), _done()]


@dataclass
class _UserInput:
    kind: str
    text: str | None = None
    action: str | None = None


def _mk_session() -> Session:
    return Session(count_tokens=lambda text: max(1, len(text.split())))


def _budget() -> Budget:
    return Budget()


_COMMON_KWARGS = dict(
    model_writes_search=False,
    rewrite_on_weak_evidence=False,
    rewrite_on_followup=False,
    host_topic_gate=False,
    child_safe_body_topics=False,
    no_specifics_without_source=False,
)


# ---------------------------------------------------------------------------
# 1. Trigger fires: route "clarify", no forced search, pending set, logged.
# ---------------------------------------------------------------------------


def test_trigger_fires_and_asks_did_you_mean():
    research = FakeResearchEngine(
        by_query={
            "what is a ardweeno": _Response(
                passages=[],
                assessment=_Assessment("weak"),
                unknown_terms=["ardweeno"],
            ),
            "Arduino": _Response(
                passages=[_Passage(label="S1", title="Arduino", text="Arduino is a board.")],
                assessment=_Assessment("strong"),
            ),
        }
    )
    llm = FakeLlmClient([_reply_script("Arduino | small single-board computer")])
    session = _mk_session()
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="what is a ardweeno"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=_budget(),
        emit=events.append,
        **_COMMON_KWARGS,
    )

    assert isinstance(result, TurnResult)
    assert result.route == "clarify"
    assert "Did you mean Arduino" in result.answer_text
    assert llm.call_count == 1  # only the clarify call, no forced search
    assert session.pending_clarify == {
        "word": "ardweeno",
        "candidate": "Arduino",
        "description": "small single-board computer",
        "message": "what is a ardweeno",
        "round": 1,
    }
    rendered = session.log.render()
    assert any(
        m.get("role") == "assistant" and "Did you mean Arduino" in (m.get("content") or "")
        for m in rendered
    )


# ---------------------------------------------------------------------------
# 2. "yes" -> normal path runs on the corrected text.
# ---------------------------------------------------------------------------


def test_yes_runs_normal_turn_on_corrected_text():
    research = FakeResearchEngine(
        by_query={
            "what is a Arduino": _Response(
                passages=[_Passage(label="S1", title="Arduino", text="Arduino is a board.")],
                assessment=_Assessment("strong"),
            ),
        }
    )
    llm = FakeLlmClient([_reply_script("Arduino is a small computer board [S1].")])
    session = _mk_session()
    session.pending_clarify = {
        "word": "ardweeno",
        "candidate": "Arduino",
        "description": "small single-board computer",
        "message": "what is a ardweeno",
        "round": 1,
    }
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="yes"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=_budget(),
        emit=events.append,
        **_COMMON_KWARGS,
    )

    assert "what is a Arduino" in research.calls
    assert session.pending_clarify is None
    assert result.evidence is not None
    assert result.evidence.get("clarify") == {"resolved": "Arduino"}
    rendered = session.log.render()
    assert any(
        m.get("role") == "user" and m.get("content") == "what is a Arduino" for m in rendered
    )


# ---------------------------------------------------------------------------
# 3. "no" -> describe prompt, round incremented.
# ---------------------------------------------------------------------------


def test_no_asks_to_describe_without_spending_a_round():
    research = FakeResearchEngine()
    llm = FakeLlmClient([])  # no LLM call expected
    session = _mk_session()
    session.pending_clarify = {
        "word": "ardweeno",
        "candidate": "Arduino",
        "description": "small single-board computer",
        "message": "what is a ardweeno",
        "round": 1,
    }
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="no"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=_budget(),
        emit=events.append,
        **_COMMON_KWARGS,
    )

    assert result.route == "clarify"
    assert 'Tell me what a "ardweeno" is' in result.answer_text
    assert session.pending_clarify["round"] == 1
    assert session.pending_clarify["asked_describe"] is True
    assert llm.call_count == 0
    assert research.call_count == 0


# ---------------------------------------------------------------------------
# 4. "no" twice -> falls back to the normal path.
# ---------------------------------------------------------------------------


def test_no_twice_falls_back_to_normal_path():
    research = FakeResearchEngine(
        by_query={
            "what is a ardweeno": _Response(
                passages=[],
                assessment=_Assessment("empty"),
                unknown_terms=["ardweeno"],
            ),
        }
    )
    llm = FakeLlmClient(
        [
            # Only the normal-turn final answer: after giving up, the
            # fallback turn must NOT re-trigger the clarify question for
            # the same word (that would ask "Did you mean Arduino?" again).
            _reply_script("I couldn't find that in the library."),
        ]
    )
    session = _mk_session()
    session.pending_clarify = {
        "word": "ardweeno",
        "candidate": "Arduino",
        "description": "small single-board computer",
        "message": "what is a ardweeno",
        "round": 2,
    }
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="no"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=_budget(),
        emit=events.append,
        **_COMMON_KWARGS,
    )

    assert session.pending_clarify is None
    assert result.route != "clarify"
    assert "what is a ardweeno" in research.calls


# ---------------------------------------------------------------------------
# 5. Gate failure -> normal path, no clarify.
# ---------------------------------------------------------------------------


def test_no_to_the_describe_prompt_gives_up_to_normal_path():
    research = FakeResearchEngine(
        by_query={
            "what is a ardweeno": _Response(passages=[], assessment=_Assessment("empty")),
        }
    )
    llm = FakeLlmClient([_reply_script("I could not find that.")])
    session = _mk_session()
    session.pending_clarify = {
        "word": "ardweeno",
        "candidate": "Arduino",
        "description": "small single-board computer",
        "message": "what is a ardweeno",
        "round": 1,
        "asked_describe": True,
    }
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="no"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=_budget(),
        emit=events.append,
        **_COMMON_KWARGS,
    )

    assert session.pending_clarify is None
    assert result.route != "clarify"
    assert "what is a ardweeno" in research.calls


def test_gate_failure_falls_through_to_normal_path():
    research = FakeResearchEngine(
        by_query={
            "what is a ardweeno": _Response(
                passages=[],
                assessment=_Assessment("weak"),
                unknown_terms=["ardweeno"],
            ),
            # "Arduino" gate lookup intentionally returns nothing matching.
            "Arduino": _Response(passages=[], assessment=_Assessment("empty")),
        }
    )
    llm = FakeLlmClient(
        [
            _reply_script("Arduino | small single-board computer"),  # framing 1
            _reply_script("Arduino | small single-board computer"),  # framing 2 (bare word)
            _reply_script("I couldn't find that in the library."),
        ]
    )
    session = _mk_session()
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="what is a ardweeno"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=_budget(),
        emit=events.append,
        **_COMMON_KWARGS,
    )

    assert result.route != "clarify"
    assert session.pending_clarify is None
    assert "Arduino" in research.calls
    # Both framings were tried, each in a FIXED context: no system prompt,
    # no lesson log, one user message (measured reason in agent_loop.py).
    first, second = llm.calls[0], llm.calls[1]
    assert [m["role"] for m in first] == ["user"]
    assert [m["role"] for m in second] == ["user"]
    assert first[0]["content"] != second[0]["content"]
    assert "what is a ardweeno" in first[0]["content"]
    assert "'ardweeno'" in second[0]["content"]


# ---------------------------------------------------------------------------
# 6. Setting off -> normal path, exactly today's behaviour.
# ---------------------------------------------------------------------------


def test_setting_off_never_triggers_clarify():
    research = FakeResearchEngine(
        by_query={
            "what is a ardweeno": _Response(
                passages=[],
                assessment=_Assessment("weak"),
                unknown_terms=["ardweeno"],
            ),
        }
    )
    llm = FakeLlmClient([_reply_script("I couldn't find that in the library.")])
    session = _mk_session()
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="what is a ardweeno"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=_budget(),
        emit=events.append,
        clarify_unknown_words=False,
        **_COMMON_KWARGS,
    )

    assert result.route != "clarify"
    assert llm.call_count == 1
    assert research.call_count == 1


# ---------------------------------------------------------------------------
# 7. Unrelated reply while pending -> re-run clarify LLM with the
#    "you asked ... they answered" note; no new gated candidate -> pending
#    cleared, normal turn runs on the reply text.
# ---------------------------------------------------------------------------


def test_description_pointing_at_same_candidate_reoffers_it_and_keeps_pending():
    # Live (Ling, 2026-09-21): "no" -> "it is a little computer you plug
    # lights into" -> the clarify call names Arduino again. Re-offer it
    # once (cheap, keeps pending so "yes" resolves) instead of a ~35s
    # search on the description that ends in "not found".
    research = FakeResearchEngine(by_query={})
    llm = FakeLlmClient([_reply_script("Arduino | small single-board computer")])
    session = _mk_session()
    session.pending_clarify = {
        "word": "ardweeno",
        "candidate": "Arduino",
        "description": "small single-board computer",
        "message": "what is a ardweeno",
        "round": 1,
    }
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="it is a little computer you plug lights into"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=_budget(),
        emit=events.append,
        **_COMMON_KWARGS,
    )

    assert result.route == "clarify"
    assert "still Arduino" in result.answer_text
    assert session.pending_clarify is not None
    assert session.pending_clarify["candidate"] == "Arduino"
    assert session.pending_clarify["round"] == 2
    assert research.calls == []
    first_call_note = llm.calls[0][-1]["content"]
    assert "they answered" in first_call_note
    assert '"it is a little computer you plug lights into"' in first_call_note


def test_short_retyped_word_with_no_new_candidate_becomes_a_fresh_turn():
    research = FakeResearchEngine(
        by_query={
            "arduino": _Response(
                passages=[_Passage(label="S1", title="Arduino", text="Arduino is a board.")],
                assessment=_Assessment("strong"),
            ),
        }
    )
    llm = FakeLlmClient(
        [
            _reply_script("Arduino | small single-board computer"),
            _reply_script("Arduino is a board [S1]."),
        ]
    )
    session = _mk_session()
    session.pending_clarify = {
        "word": "ardweeno",
        "candidate": "Arduino",
        "description": "small single-board computer",
        "message": "what is a ardweeno",
        "round": 1,
    }
    events: list = []

    result = run_turn(
        session,
        _UserInput(kind="text", text="arduino"),
        llm=llm,
        research_engine=research,
        calc=FakeCalc(),
        budget=_budget(),
        emit=events.append,
        **_COMMON_KWARGS,
    )

    assert session.pending_clarify is None
    assert result.route != "clarify"
    assert "arduino" in research.calls

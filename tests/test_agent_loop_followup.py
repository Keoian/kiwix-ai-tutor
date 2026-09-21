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
import re
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
    return _Response(
        passages=[
            _Passage(
                label=f"S{n}",
                passage_id=f"pid-{n}",
                text=(
                    f"Passage number {n}: the moons, planets, and asteroids of "
                    "the solar system all orbit the sun. "
                )
                * 10,
            )
        ]
    )


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
        model_writes_search=False,
    )

    result = run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=False,
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
    # The rewrite's own evidence (originally labelled "S3" by its own
    # retrieval call, relabelled "S1" -- sequential, displayed-order
    # labelling within this packet) leads; the raw turn-2 pre-search's
    # passage (originally "S2", relabelled "S2" here too) is only ever
    # used as backfill behind it, never ahead.
    assert "[S1] Passage number 3" in joined
    assert joined.index("[S1] Passage number 3") < joined.index("[S2] Passage number 2")
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
        model_writes_search=False,
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
        model_writes_search=False,
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
        model_writes_search=False,
    )

    assert result.status == "ok"
    assert llm.call_count == 2  # no extra forced-rewrite round
    rendered = session.log.render()
    assert "Host note" not in rendered[4]["content"]


def test_followup_forced_round_pastes_pointer_for_held_passage():
    """The forced-rewrite round's own tool-result text must go through the
    same held-id/pointer bookkeeping as a plain evidence packet
    (docs/passage_reuse.md): turn 1 pastes pid-1 in full; turn 2's forced
    round merges the raw pre-search's pid-1 (as backfill) behind the
    rewrite's own new pid-2 -- pid-1 must come back as a pointer, not a
    second full paste, and pid-2 (new) must be recorded as held so a
    third turn reusing it also gets a pointer."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets"]}),
            _final("There are also moons [S2]."),
            _tool_call("research", {"queries": ["asteroids too"]}),
            _final("Asteroids exist too [S2]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1),  # turn 1 raw pre-search -> pid-1, held after turn 1
            _strong_response(1),  # turn 2 raw pre-search (backfill only) -> pid-1 again
            _strong_response(2),  # turn 2 rewrite's own query -> pid-2 (new)
            _strong_response(2),  # turn 3 raw pre-search (backfill only) -> pid-2 again
            _strong_response(2),  # turn 3 rewrite's own query -> pid-2 again (held by now)
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
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=False,
    )

    assert "pid-1" in session.log.held_ids()
    assert "pid-2" in session.log.held_ids()

    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    turn2_tool_text = tool_messages[-1].get("content") or ""
    full_pid1_text = (
        "Passage number 1: the moons, planets, and asteroids of "
        "the solar system all orbit the sun. "
    ) * 10
    assert full_pid1_text not in turn2_tool_text
    assert "already shown above" in turn2_tool_text

    run_turn(
        session,
        _UserInput(kind="text", text="anything else"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=False,
    )

    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    turn3_tool_text = tool_messages[-1].get("content") or ""
    full_pid2_text = (
        "Passage number 2: the moons, planets, and asteroids of "
        "the solar system all orbit the sun. "
    ) * 10
    # pid-2's full text was pasted once already (turn 2's rewrite result);
    # turn 3's forced round must not paste it again in full.
    assert full_pid2_text not in turn3_tool_text
    assert "already shown above" in turn3_tool_text


def test_followup_forced_round_off_setting_pastes_full_text_every_time():
    """``reuse_prior_passages=False`` must reproduce the old
    byte-for-byte behaviour for the forced-rewrite round too: no pointer
    substitution, the same passage's full text pasted again."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets"]}),
            _final("There are also moons [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1),  # turn 1 raw pre-search -> pid-1
            _strong_response(1),  # turn 2 raw pre-search (backfill only) -> pid-1
            _strong_response(1),  # turn 2 rewrite's own query -> pid-1 again
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
        reuse_prior_passages=False,
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        reuse_prior_passages=False,
        model_writes_search=False,
    )

    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    turn2_tool_text = tool_messages[-1].get("content") or ""
    full_pid1_text = (
        "Passage number 1: the moons, planets, and asteroids of "
        "the solar system all orbit the sun. "
    ) * 10
    assert full_pid1_text in turn2_tool_text
    assert "already shown above" not in turn2_tool_text


def test_no_passage_pasted_twice_in_full_within_one_followup_turn():
    """A passage must never be pasted in full twice within a single turn
    -- specifically, the raw pre-search packet's passages (used only as
    backfill on a follow-up turn, never appended to the log directly) must
    not end up duplicated against the forced round's own tool-result
    text."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets"]}),
            _final("There are also moons [S1][S2]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1),  # turn 1 raw pre-search -> pid-1
            # turn 2 raw pre-search (backfill) returns the SAME pid-1 the
            # rewrite's own query also happens to return -- the merge must
            # not duplicate it, and it must not be pasted twice this turn.
            _strong_response(1),
            _strong_response(1),
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
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=False,
    )

    rendered = session.log.render()
    full_pid1_text = (
        "Passage number 1: the moons, planets, and asteroids of "
        "the solar system all orbit the sun. "
    ) * 10
    turn2_messages = rendered[4:]  # user(turn2) onward, per the existing index convention
    full_paste_count = sum(
        1 for m in turn2_messages if full_pid1_text in (m.get("content") or "")
    )
    assert full_paste_count == 0  # pid-1 was already held from turn 1 -> pointer only


def test_concise_followup_note_default_is_off_uses_plain_directness_note():
    """docs/followup_answer_shape.md, "Iteration 2 (negative result)":
    ``concise_followup_note`` defaults to False (both ``run_turn``'s own
    keyword default and ``AppConfig.concise_followup_note``) after the
    stronger note was measured to over-correct (a spurious "Yes," tic on
    open follow-up questions). With no override, the plain, pre-existing
    ``_FOLLOWUP_DIRECTNESS_NOTE`` wording lands in the forced round's
    tool result on turn >= 2, never on turn 1."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets in the solar system"]}),
            _final("There are also moons and asteroids [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [_strong_response(1), _strong_response(2), _strong_response(3)]
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
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=False,
    )

    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "do not restate points" not in joined
    assert "ONLY what is new" not in joined
    assert "then explain using the sources above" in joined
    # Turn 1 has no forced round at all, so no note of either kind lands.
    assert "Host note" not in (rendered[1].get("content") or "")


def test_concise_followup_note_true_opts_into_stronger_wording():
    """``concise_followup_note=True`` still opts into the stronger
    "only what is new" / "1-3 sentences for yes/no" wording (kept
    available for further experimentation per docs/followup_answer_shape.md,
    not the default)."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets in the solar system"]}),
            _final("There are also moons and asteroids [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [_strong_response(1), _strong_response(2), _strong_response(3)]
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
        concise_followup_note=True,
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        concise_followup_note=True,
        model_writes_search=False,
    )

    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "do not restate points" in joined
    assert "ONLY what is new" in joined


def test_restate_question_last_off_reproduces_plain_bytes():
    """``restate_question_last=False`` reproduces the exact same
    tool-result bytes as before this setting existed (the default is
    now True per docs/followup_answer_shape.md "Decision (orchestrator)")."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets in the solar system"]}),
            _final("There are also moons and asteroids [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [_strong_response(1), _strong_response(2), _strong_response(3)]
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
        restate_question_last=False,
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        restate_question_last=False,
        model_writes_search=False,
    )

    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "The student is now asking" not in joined


def test_restate_question_last_r1_lands_on_followup_turn_only():
    """R1: turn >= 2's forced-rewrite round appends a restatement line
    using the ORIGINAL user text and the model's own rewritten query
    from that same round, as the last thing in the tool result. Turn 1
    (no forced round) never gets it."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets in the solar system"]}),
            _final("There are also moons and asteroids [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [_strong_response(1), _strong_response(2), _strong_response(3)]
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
        restate_question_last=True,
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        restate_question_last=True,
        model_writes_search=False,
    )

    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    assert "The student is now asking" not in (tool_messages[0].get("content") or "")
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert (
        'The student is now asking: "what else is in the solar system" '
        '(meaning: moons and planets in the solar system). Answer THIS question.'
    ) in joined
    # It is the LAST thing appended to that tool result.
    last_tool_content = tool_messages[-1]["content"]
    assert last_tool_content.rstrip().endswith("Answer THIS question.")
    assert "If it is a yes/no question start with Yes or No" not in joined


def test_restate_question_r2_adds_instruction_sentence():
    """R2: same restatement line, plus one extra instruction sentence
    right after it -- only when ``restate_question_instruction`` is also
    True (meaningless on its own without ``restate_question_last``)."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets in the solar system"]}),
            _final("There are also moons and asteroids [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [_strong_response(1), _strong_response(2), _strong_response(3)]
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
        restate_question_last=True,
        restate_question_instruction=True,
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        restate_question_last=True,
        restate_question_instruction=True,
        model_writes_search=False,
    )

    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    last_tool_content = tool_messages[-1]["content"]
    assert (
        "Answer THIS question. If it is a yes/no question start with Yes or No"
        in last_tool_content
    )
    assert last_tool_content.rstrip().endswith("do not repeat your earlier answer.")


def test_forced_rewrite_round_labels_are_unique_and_sequential():
    """A forced-rewrite round's merged evidence packet (held/backfilled
    pointer lines plus newly-fetched passages) must never repeat an
    ``[S#]`` label within the same packet, and labels must be assigned
    S1..Sn sequentially in the order the passages are actually displayed.

    Reproduces the real bug: the backfill passage (already held, so it
    renders as a pointer line under its ORIGINAL label) and the rewrite's
    own freshly-fetched passage are each independently labelled by their
    own upstream retrieval call, which always numbers its own results
    starting from S1 -- so both can land as "S1" in the same packet
    before this is fixed."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _final("The solar system has eight planets [S1]."),
            _tool_call("research", {"queries": ["moons and planets in the solar system"]}),
            _final("There are also moons and asteroids [S1]."),
        ]
    )
    # Turn 1: pid-1 fetched and labelled "S1" by its own (independent)
    # retrieval call -- held after turn 1.
    turn1 = _strong_response(1)
    # Turn 2 raw pre-search (backfill only): re-finds pid-1, again
    # independently labelled "S1" by this call's own retrieval numbering.
    turn2_backfill = _strong_response(1)
    # Turn 2 rewrite's own query: a brand-new passage (pid-2) that this
    # independent retrieval call ALSO labels "S1" (it has no idea pid-1
    # already claimed that label in this lesson) -- the real-world
    # collision.
    turn2_rewrite = _Response(
        passages=[
            _Passage(
                label="S1",
                passage_id="pid-2",
                text=(
                    "Passage number 2: the moons, planets, and asteroids of "
                    "the solar system all orbit the sun. "
                )
                * 10,
            )
        ]
    )
    research = ScriptedResearchEngine([turn1, turn2_backfill, turn2_rewrite])
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="what is the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=False,
    )
    run_turn(
        session,
        _UserInput(kind="text", text="what else is in the solar system"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=False,
    )

    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    turn2_tool_text = tool_messages[-1].get("content") or ""

    labels_in_order = re.findall(r"^\[(S\d+)\]", turn2_tool_text, flags=re.MULTILINE)
    assert len(labels_in_order) >= 2, turn2_tool_text
    assert len(labels_in_order) == len(set(labels_in_order)), (
        f"duplicate [S#] label within one packet: {labels_in_order}\n{turn2_tool_text}"
    )
    assert labels_in_order == [f"S{i + 1}" for i in range(len(labels_in_order))], (
        f"labels not sequential in displayed order: {labels_in_order}"
    )

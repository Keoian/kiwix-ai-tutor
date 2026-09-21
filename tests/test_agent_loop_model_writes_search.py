"""Tests for ``app.model_writes_search`` (default True, adopted -- see
docs/model_writes_search_measure.md and docs/rewrite_on_weak_evidence.md,
"Model writes every search": the model
crafts the library search on EVERY turn, including turn 1, using
short/title-like queries rather than the raw student text or a
full-sentence rewrite. Mirrors tests/test_agent_loop_followup.py's
pattern (real Session/PromptLog, scripted fake LLM, fake research/calc,
no live LLM).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tutor.app.agent_loop import _clip_model_written_queries, run_turn
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
    title: str = "Titin"
    path: str = "A/Titin"
    text: str = "Titin is the largest known protein, made of many amino acids. " * 10
    kind: str = "article"


@dataclass
class _Response:
    status: str = "ok"
    passages: list = field(default_factory=list)
    assessment: object = None
    corrected_terms: dict = field(default_factory=dict)


class ScriptedResearchEngine:
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
    session = Session(count_tokens=_count_tokens, subject_hint="biology")
    return session, Budget()


def _strong_response(n: int, title: str = "Titin") -> _Response:
    return _Response(
        passages=[
            _Passage(
                label=f"S{n}",
                passage_id=f"pid-{n}",
                title=title,
                path=f"A/{title.replace(' ', '_')}",
                text=(f"Passage number {n}: {title} is a large biological molecule. ") * 10,
            )
        ]
    )


def test_off_reproduces_old_bytes_on_turn_one():
    """``model_writes_search=False`` -- turn 1 with strong raw evidence
    never fires a forced round (old, byte-identical behaviour). Default
    is now True (docs/model_writes_search_measure.md "Decision"), so this
    test passes the flag explicitly rather than relying on the default."""
    session, budget = _mk_session()
    llm = FakeLlmClient([_final("Titin is the largest protein [S1].")])
    research = ScriptedResearchEngine([_strong_response(1)])
    calc = FakeCalc()

    result = run_turn(
        session,
        # Chosen so the raw pre-search's own assessment is "strong" (its
        # words overlap the passage text), isolating this test to
        # "no forced round fires" rather than the pre-existing
        # weak-evidence rewrite mechanism.
        _UserInput(kind="text", text="What is Titin"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=False,
    )

    assert result.status == "ok"
    assert llm.call_count == 1  # no forced round
    rendered = session.log.render()
    assert "Host note" not in rendered[1]["content"]


def test_on_fires_forced_round_on_turn_one():
    """With the setting on, turn 1 gets a host note and a forced research
    tool-call round BEFORE the model answers, even though this is the
    very first turn of the lesson."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Titin", "Largest protein"]}),
            _final("Titin is the largest known protein [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),  # turn 1 raw pre-search (backfill only)
            _strong_response(2, title="Titin"),  # model query "Titin"
            _strong_response(3, title="Largest protein"),  # model query "Largest protein"
        ]
    )
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="What's the largest molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )

    assert result.status == "ok"
    assert llm.tool_choice_calls[0] == {"type": "function", "function": {"name": "research"}}
    rendered = session.log.render()
    assert "Host note" in rendered[1]["content"]
    assert "encyclopedia article titles" in rendered[1]["content"]
    assert result.evidence["rewritten_queries"] == ["Titin", "Largest protein"]
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "Searched for: Titin, Largest protein" in joined
    # The model's own queries' results lead; the raw pre-search's
    # wrong-topic "Molecule" passage only ever backfills remaining slots.
    assert joined.index("[S1]") < joined.index("Passage number 1")


def test_on_replaces_followup_round_on_turn_two_not_both():
    """On turn >= 2, ``model_writes_search`` replaces the follow-up
    rewrite round rather than running both -- exactly one forced round,
    i.e. exactly 2 LLM calls that turn (forced tool call + final answer),
    not 3."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Titin"]}),
            _final("Titin is the largest known protein [S1]."),
            _tool_call("research", {"queries": ["DNA"]}),
            _final("DNA is a different molecule [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),
            _strong_response(2, title="Titin"),
            _strong_response(3, title="Molecule"),
            _strong_response(4, title="DNA"),
        ]
    )
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="What's the largest molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )
    calls_before_turn2 = llm.call_count

    result = run_turn(
        session,
        _UserInput(kind="text", text="Is it a molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )

    assert result.status == "ok"
    assert llm.call_count - calls_before_turn2 == 2  # exactly one forced round, not two


def test_clip_helper_strips_quotes_question_marks_and_extra_words():
    """Unit-level: at most 3 queries; each stripped of surrounding quotes
    and a trailing '?', clipped to 8 words; empties dropped."""
    long_query = " ".join(f"word{i}" for i in range(20))
    cleaned = _clip_model_written_queries(
        ['"Titin"', "Largest protein?", long_query, "extra fourth query dropped"]
    )
    assert cleaned == [
        "Titin",
        "Largest protein",
        " ".join(f"word{i}" for i in range(8)),
    ]


def test_clip_helper_drops_unusable_entries():
    assert _clip_model_written_queries(['""', "   ", "?", ""]) == []
    assert _clip_model_written_queries([]) == []


def test_up_to_three_queries_are_all_searched_and_merged():
    """The tool schema itself already caps ``queries`` at 3 items (a
    4th-item call is rejected as invalid and never reaches the host's own
    clipping); this test covers the in-range case: all 3 of the model's
    queries are searched via ``research_many`` and merged, model-first."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call(
                "research",
                {"queries": ["Titin", "Largest protein", "extra third"]},
            ),
            _final("Titin is the largest known protein [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),
            _strong_response(2, title="Titin"),
            _strong_response(3, title="Largest protein"),
            _strong_response(4, title="Extra third"),
        ]
    )
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="What's the largest molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )

    assert result.status == "ok"
    assert result.evidence["rewritten_queries"] == ["Titin", "Largest protein", "extra third"]
    # 1 raw pre-search + 3 of the model's own queries.
    assert research.call_count == 4


def test_fallback_to_old_behaviour_when_model_output_unusable():
    """If the model's queries clean down to nothing usable (e.g. only
    whitespace/quote characters), the forced round falls back to today's
    not-found behaviour for that round rather than crashing or searching
    on garbage -- the empty evidence then still gets exactly one more
    (uncapped, non-model-written) weak-evidence rewrite round, per the
    existing ``rewrite_on_weak_evidence`` mechanism (unchanged by this
    setting)."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["\"\"", "   ", "?"]}),
            _tool_call("research", {"queries": ["still nothing useful"]}),
            _final("I could not find that in the library."),
        ]
    )
    # Only the turn's raw pre-search runs (backfill only) -- clipping
    # empties the model's queries before any research_many call is made
    # for the first (model-written) round; the second, uncapped
    # weak-evidence round then makes its own real research call.
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),
            _strong_response(2, title="Molecule again"),
        ]
    )
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="What's the largest molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )

    assert result.status == "ok"
    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    # The first (model-written) forced round produced nothing usable, so
    # it fell straight to the not-found tool text with no queries listed.
    assert "No good match was found" in joined


def test_restate_meaning_uses_model_question_not_keywords():
    """STEP 1/2 regression test: in model_writes_search mode, the forced
    round's model tool call can carry an optional standalone ``question``
    string alongside its keyword ``queries``. The restate line's
    "(meaning: ...)" must show that standalone question, NOT the keyword
    queries joined -- today (before this fix) it falls back to
    ``rewritten_queries[0]``, e.g. "monomer", losing the student's actual
    referent (e.g. "tires")."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Titin"]}),
            _final("Titin is the largest known protein [S1]."),
            _tool_call(
                "research",
                {
                    "question": "Is a tire a single molecule?",
                    "queries": ["Monomer"],
                },
            ),
            _final("No, a tire is not a single molecule [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),
            _strong_response(2, title="Titin"),
            _strong_response(3, title="Molecule"),
            _strong_response(4, title="Monomer"),
        ]
    )
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="What's the longest molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )
    result = run_turn(
        session,
        _UserInput(kind="text", text="They're a single molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )

    assert result.status == "ok"
    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert '(meaning: Is a tire a single molecule?)' in joined
    assert "(meaning: Monomer" not in joined


def test_restate_meaning_omitted_when_model_writes_no_question():
    """When the model's forced tool call has no ``question`` field, the
    restate line must NOT fall back to showing the keyword queries as the
    "meaning" -- it should omit the "(meaning: ...)" clause entirely."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Titin"]}),
            _final("Titin is the largest known protein [S1]."),
            _tool_call("research", {"queries": ["Monomer"]}),
            _final("No, a monomer is not a single molecule [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),
            _strong_response(2, title="Titin"),
            _strong_response(3, title="Molecule"),
            _strong_response(4, title="Monomer"),
        ]
    )
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="What's the longest molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )
    result = run_turn(
        session,
        _UserInput(kind="text", text="They're a single molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )

    assert result.status == "ok"
    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "The student is now asking:" in joined
    assert "(meaning:" not in joined


def test_restate_line_included_on_turn_one_when_question_present():
    """New behaviour: the restate line now also fires on turn 1 when the
    model's forced round supplies a ``question`` (previously it only ever
    fired on turn >= 2, since turn 1 has no prior referent to resolve)."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call(
                "research",
                {"question": "What is the largest molecule?", "queries": ["Titin"]},
            ),
            _final("Titin is the largest known protein [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),
            _strong_response(2, title="Titin"),
        ]
    )
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="What's the largest molecule?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
    )

    assert result.status == "ok"
    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "(meaning: What is the largest molecule?)" in joined


def test_clip_model_written_question_clips_to_30_words_and_single_line():
    from tutor.app.agent_loop import _clip_model_written_question

    long_question = "Is this " + " ".join(f"word{i}" for i in range(40)) + " a molecule?"
    clipped = _clip_model_written_question(long_question)
    assert clipped is not None
    assert len(clipped.split()) <= 30

    multiline = "Is a tire\na single molecule?"
    assert _clip_model_written_question(multiline) == "Is a tire a single molecule?"

    assert _clip_model_written_question(None) is None
    assert _clip_model_written_question("   ") is None

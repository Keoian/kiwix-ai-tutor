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
        self.tools_calls: list[object] = []
        self.response_format_calls: list[object] = []

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
        self.tools_calls.append(tools)
        self.response_format_calls.append(response_format)
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
    assert "[Host:" not in rendered[1]["content"]


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
    assert "[Host:" in rendered[1]["content"]
    assert "How to call research" in rendered[1]["content"]
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


def test_forced_call_schema_has_needs_search_first_queries_bounded():
    """2026-09-21 cache fix: the forced round's own request must send the
    EXACT SAME ``tools`` list as every other call in the turn (the answer
    round, voluntary rounds) -- a different ``tools`` payload gives
    llama-server a different prompt prefix and defeats the KV cache for
    the rest of the turn. So there is no longer a separate
    ``FORCED_RESEARCH_TOOL`` schema; ``FORCED_RESEARCH_TOOL`` is now just
    an alias for the single ``RESEARCH_TOOL`` used everywhere, with
    ``needs_search`` first (a hint for a grammar-constrained server),
    nothing ``required`` (so a voluntary call can still omit it), and
    ``queries`` bounded 0-3."""
    from tutor.tools.schemas import FORCED_RESEARCH_TOOL, RESEARCH_TOOL, validate_tool_call

    assert FORCED_RESEARCH_TOOL is RESEARCH_TOOL
    params = RESEARCH_TOOL["function"]["parameters"]
    prop_names = list(params["properties"].keys())
    assert prop_names.index("needs_search") < prop_names.index("question")
    assert prop_names.index("question") < prop_names.index("queries")
    assert params["properties"]["needs_search"]["type"] == "boolean"
    assert params["properties"]["queries"]["minItems"] == 0
    assert params["properties"]["queries"]["maxItems"] == 3

    # Nothing is schema-``required``: a voluntary call can still omit
    # ``question``/``needs_search`` entirely.
    assert params["required"] == []
    result = validate_tool_call("research", json.dumps({"queries": ["Titin"]}))
    assert result.ok is True


def test_forced_call_sends_the_same_tools_list_as_the_answer_round():
    """End-to-end: the ``tools`` kwarg on the forced round's own
    ``stream_chat`` call is byte-identical (same object) to the one the
    final answer round sends -- the cache-defeating bug this fixes."""
    from tutor.tools.schemas import TOOLS

    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call(
                "research",
                {"question": "What is the longest molecule?", "queries": ["Titin"]},
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

    forced_call_tools = llm.tools_calls[0]
    answer_call_tools = llm.tools_calls[1]
    assert forced_call_tools == TOOLS
    assert forced_call_tools is answer_call_tools


# ---------------------------------------------------------------------------
# app.model_may_skip_search (default True) -- see
# docs/rewrite_on_weak_evidence.md, "Model may skip the search".
# ---------------------------------------------------------------------------


def test_needs_search_false_skips_search_and_appends_no_search_result():
    """When the model's forced call sets needs_search=false, no search
    runs at all -- research_engine.research_many/research is never called
    a second time (only the turn's raw pre-search, which is discarded),
    and the tool result tells the model to answer conversationally."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"needs_search": False, "queries": []}),
            _final("You are asking about your own speed -- not in the library!"),
        ]
    )
    research = ScriptedResearchEngine([_strong_response(1, title="Usain Bolt")])
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="How fast am I?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
        model_may_skip_search=True,
    )

    assert result.status == "ok"
    assert research.call_count == 1  # only the raw pre-search; no real search
    assert result.evidence["level_after"] == "skipped"
    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "No library search needed for this message" in joined
    assert "Searched for:" not in joined


def test_needs_search_false_captured_via_statuses():
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"needs_search": False, "queries": []}),
            _final("Sure thing."),
        ]
    )
    research = ScriptedResearchEngine([_strong_response(1)])
    calc = FakeCalc()
    statuses = []

    def _collect(e):
        if isinstance(e, dict) and e.get("kind") == "status":
            statuses.append(e)

    run_turn(
        session,
        _UserInput(kind="text", text="lol ok thanks"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=_collect,
        model_writes_search=True,
        model_may_skip_search=True,
    )

    details = [s.get("detail") for s in statuses]
    assert "No need to look this up..." in details
    assert not any(d and d.startswith("Searching the library for") for d in details)


def test_model_may_skip_search_off_forces_search_even_with_needs_search_false():
    """``app.model_may_skip_search=False`` reproduces today's bytes exactly
    -- a needs_search=false argument is simply ignored and the (now
    empty) queries fall back to the existing not-found behaviour, never
    a 'skipped' evidence level."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"needs_search": False, "queries": []}),
            _tool_call("research", {"needs_search": False, "queries": ["Usain Bolt"]}),
            _final("I could not find that."),
        ]
    )
    research = ScriptedResearchEngine([_strong_response(1), _strong_response(2)])
    calc = FakeCalc()

    result = run_turn(
        session,
        _UserInput(kind="text", text="How fast am I?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
        model_may_skip_search=False,
    )

    assert result.status == "ok"
    assert result.evidence["level_after"] != "skipped"


def test_needs_search_true_with_empty_queries_falls_back_to_not_found():
    """needs_search left true (or omitted) but no usable queries -- today's
    existing fallback behaviour, unaffected by app.model_may_skip_search."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"needs_search": True, "queries": []}),
            _tool_call("research", {"queries": ["still nothing useful"]}),
            _final("I could not find that in the library."),
        ]
    )
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
        model_may_skip_search=True,
    )

    assert result.status == "ok"
    assert result.evidence["level_after"] != "skipped"
    rendered = session.log.render()
    tool_messages = [m for m in rendered if m["role"] == "tool"]
    joined = "\n".join(m.get("content") or "" for m in tool_messages)
    assert "No good match was found" in joined


def test_weak_evidence_second_round_never_fires_after_skip():
    """Once a turn's evidence level is 'skipped', the weak-evidence extra
    round must never fire (it only fires on level in ('weak', 'empty'))."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"needs_search": False, "queries": []}),
            _final("Sure."),
        ]
    )
    research = ScriptedResearchEngine([_strong_response(1)])
    calc = FakeCalc()

    run_turn(
        session,
        _UserInput(kind="text", text="thanks!"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=True,
        model_may_skip_search=True,
    )

    # Only 2 LLM calls total: the forced round + the final answer -- no
    # second (weak-evidence) forced round.
    assert llm.call_count == 2


def test_forced_schema_response_format_fallback_includes_needs_search():
    from tutor.app.agent_loop import _forced_research_tool_call

    class _RejectingToolChoiceLlm:
        def __init__(self):
            self.response_format_calls = []

        def stream_chat(self, messages, *, tool_choice=None, response_format=None, **kwargs):
            if tool_choice is not None:
                yield StreamEvent(kind="error", error="tool_choice unsupported")
                return
            self.response_format_calls.append(response_format)
            yield StreamEvent(
                kind="token",
                text=json.dumps(
                    {"needs_search": False, "question": "thanks", "queries": []}
                ),
            )
            yield StreamEvent(kind="done", finish_reason="stop")

    llm = _RejectingToolChoiceLlm()
    forced = _forced_research_tool_call(llm, [{"role": "user", "content": "hi"}], cancel=None)
    assert forced is not None
    _, queries, _arguments_json, _question, needs_search = forced
    assert queries == []
    assert needs_search is False
    schema = llm.response_format_calls[0]["json_schema"]["schema"]
    assert schema["required"] == ["needs_search", "question", "queries"]
    assert schema["properties"]["queries"]["minItems"] == 0


# ---------------------------------------------------------------------------
# Truncated forced-call JSON salvage (2026-09-21 live measurement on Ling
# 3.0 Tiny: needs_search + a full-sentence question + 3 queries can exceed
# the forced round's own output cap, truncating the JSON mid-argument).
# ---------------------------------------------------------------------------


def _truncated_tool_call(raw_json: str, call_id="call_1"):
    return [
        StreamEvent(kind="tool_call", id=call_id, name="research", arguments_json=raw_json),
        StreamEvent(kind="done", finish_reason="length", usage={}),
    ]


def test_truncated_forced_json_is_salvaged_when_possible():
    from tutor.app.agent_loop import _forced_research_tool_call

    truncated = (
        '{"needs_search": true, "question": "What is titin", '
        '"queries": ["Titin", "Larg'
    )
    llm = FakeLlmClient([_truncated_tool_call(truncated)])
    diagnostics: dict = {}
    forced = _forced_research_tool_call(
        llm, [{"role": "user", "content": "hi"}], cancel=None, diagnostics=diagnostics
    )
    assert forced is not None
    _, queries, _arguments_json, _question, needs_search = forced
    assert queries == ["Titin", "Larg"]
    assert needs_search is True
    assert diagnostics["truncated"] is True


def test_truncated_forced_json_with_nothing_salvageable_falls_back_to_raw_presearch():
    """When the forced call's JSON is truncated beyond any salvage, the
    round falls back to the raw pre-search's own passages (the student's
    own words) instead of reporting a dead end -- no extra LLM/search call
    is made."""
    session, budget = _mk_session()
    unsalvageable = '{"needs_search"'  # cut off before any usable content
    llm = FakeLlmClient(
        [
            _truncated_tool_call(unsalvageable),
            _final("Titin is a very long protein [S1]."),
        ]
    )
    research = ScriptedResearchEngine([_strong_response(1, title="Titin")])
    calc = FakeCalc()
    statuses = []

    def _collect(e):
        if isinstance(e, dict) and e.get("kind") == "status":
            statuses.append(e)

    result = run_turn(
        session,
        _UserInput(kind="text", text="What is titin"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=_collect,
        model_writes_search=True,
        model_may_skip_search=True,
    )

    assert result.status == "ok"
    assert research.call_count == 1  # only the raw pre-search; no wasted retry
    assert result.evidence["level_after"] == "strong"
    details = [s.get("detail") for s in statuses]
    assert any(d and "took too long" in d for d in details)


def test_every_llm_call_in_a_turn_gets_byte_identical_tools():
    """2026-09-21 cache-defeating bug: llama-server renders the ``tools``
    block near the top of the prompt, so if the forced round's call and
    the answer round's call carry different ``tools`` payloads, the whole
    lesson gets re-read from scratch on every call. Every ``stream_chat``
    call within one turn -- forced round, second (weak-evidence) round,
    answer round, and any voluntary follow-up round -- must send the
    exact same ``tools`` list."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Titin"]}),
            _tool_call("research", {"query": "Titin details"}, call_id="call_2"),
            _final("Titin is the largest known protein [S1]."),
        ]
    )
    research = ScriptedResearchEngine(
        [
            _strong_response(1, title="Molecule"),
            _strong_response(2, title="Titin"),
            _strong_response(3, title="Titin"),
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

    dumped = [json.dumps(t, sort_keys=True) for t in llm.tools_calls]
    assert len(dumped) >= 2
    assert len(set(dumped)) == 1


def test_forced_call_prefix_is_an_append_only_subset_of_the_answer_call():
    """The messages sent to the forced round's ``stream_chat`` call must be
    an exact prefix of the messages sent to the answer round's call
    (append-only, cache-safe): nothing in the earlier messages is edited
    or removed before the answer round."""
    session, budget = _mk_session()
    llm = FakeLlmClient(
        [
            _tool_call("research", {"queries": ["Titin"]}),
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

    forced_messages = llm.calls[0]
    answer_messages = llm.calls[1]
    assert answer_messages[: len(forced_messages)] == forced_messages


def test_per_turn_host_note_is_small_now_that_guidance_lives_in_system_prompt():
    """Job 1 (docs/rewrite_on_weak_evidence.md): the worked examples and
    the needs_search rule now live once in system_prompt.txt's "How to
    call research" section (read once per lesson, cached); the per-turn
    note re-read every turn must just trigger the behaviour, budgeted at
    <= 60 tokens combined (model_writes_search + model_may_skip_search)."""
    from tutor.app.agent_loop import (
        _MODEL_MAY_SKIP_SEARCH_NOTE,
        _MODEL_WRITES_SEARCH_HOST_NOTE,
        _load_default_system_text,
    )

    combined = _MODEL_WRITES_SEARCH_HOST_NOTE + _MODEL_MAY_SKIP_SEARCH_NOTE
    assert _count_tokens(combined) <= 60

    system_text = _load_default_system_text()
    assert "How to call research" in system_text
    assert "needs_search" in system_text

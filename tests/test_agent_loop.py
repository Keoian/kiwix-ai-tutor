"""RED tests for tutor.app.agent_loop.run_turn -- the six scripted lessons.

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §6 (turn flow:
receive -> route by action-vs-free-text -> pre-retrieve -> build prompt ->
inference request 1 -> tool dispatch -> inference request 2 -> finish; the
research cap message: "If the model requests a second `research` after the
cap, the host returns a tool result stating the cap is reached and the
model answers the supported portion or asks for clarification.") and
docs/plan/offline_tutor_implementation_plan.md WP-C2 (`research` cap 2
INCLUDING the host pre-retrieval, `calc` cap 4, schema validation, never
parse prose, "the six scripted lessons from 1.0").

Contract decisions / derivation of the six scripted lessons (the spec/plan
text above does not spell out a numbered list of six by name -- this is
the task brief's own enumeration, adopted verbatim as the six lessons
under test):

1. Free-text factual question: host pre-retrieves BEFORE the model is
   called at all (research call 1 of 2); evidence is appended as a
   tool-result-shaped entry; the model answers citing `[S1]`;
   `TurnResult.route == "preretrieve"`.
2. The model's first reply is itself a `research` follow-up call: it is
   executed as call 2 of 2; `route == "preretrieve+followup"`.
3. The model asks for a THIRD research (after the pre-retrieve + one
   follow-up already used the cap of 2): the host does not call the real
   research engine a third time -- it returns a cap-reached tool result,
   and the model's next reply (a "final" answer over what it already has)
   is what the turn returns. Research engine call count stays at 2.
4. A deterministic action ("simpler") never retrieves; `route == "action"`.
5. An arithmetic free-text question: the model calls `calc`; the result
   is appended and the final answer contains the computed value. The calc
   cap is 4 per turn (separate from the research cap): a fifth calc call
   in the same turn gets a cap-reached tool result instead of being run.
6. A malformed tool call (bad JSON arguments, an unknown tool name, or a
   schema violation) is never executed against the real tool; the host
   returns a validation-error tool result to the model and the turn still
   completes normally. Separately: prose that merely *looks* like a tool
   call (e.g. text containing literal `{"name": "calc", ...}`) is never
   parsed as one -- this is exercised by using a fake LLM client that
   never emits a "tool_call" StreamEvent for such text, and asserting the
   research/calc fakes were not invoked.

Everything here is a pure unit test: the LLM is a scripted FAKE client that
yields pre-programmed sequences of tutor.app.llm_client.StreamEvent per
call (one call = one `stream_chat(...)` invocation), and research/calc are
FAKE callables with call logs -- no network, no real model, no real
subprocess sandbox.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from tutor.app.agent_loop import TurnResult, run_turn
from tutor.app.llm_client import StreamEvent

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLlmClient:
    """Scripted fake: each call to stream_chat() pops the next scripted
    sequence of StreamEvents from a queue and yields them. Records every
    call's messages for order/content assertions.
    """

    def __init__(self, scripts: list[list[StreamEvent]]):
        self._scripts = list(scripts)
        self.calls: list[list[dict]] = []

    def stream_chat(self, messages, *, max_tokens=None, tools=None, cancel=None, temperature=None):
        self.calls.append(list(messages))
        if not self._scripts:
            raise AssertionError("FakeLlmClient: no more scripted responses")
        script = self._scripts.pop(0)
        for event in script:
            if cancel is not None and cancel.is_set():
                yield StreamEvent(kind="done", finish_reason="cancelled")
                return
            yield event

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeResearchEngine:
    def __init__(self, responses: list[Any] | None = None):
        self._responses = list(responses) if responses else []
        self.calls: list[dict] = []

    def research(self, query: str, *, topic_hint: str | None = None, keywords=None):
        self.calls.append({"query": query, "topic_hint": topic_hint, "keywords": keywords})
        if self._responses:
            return self._responses.pop(0)
        return _fake_response(f"S{len(self.calls)}")

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _fake_response(label: str):
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

    return _Response(status="ok", passages=[_Passage(label=label)])


class FakeCalc:
    def __init__(self):
        self.calls: list[str] = []

    def evaluate(self, expression: str, **kwargs):
        self.calls.append(expression)
        return {"ok": True, "result": "4"}

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _token(text: str) -> StreamEvent:
    return StreamEvent(kind="token", text=text)


def _done(finish_reason="stop", usage=None) -> StreamEvent:
    return StreamEvent(kind="done", finish_reason=finish_reason, usage=usage or {})


def _tool_call(name: str, arguments: dict, call_id: str = "call_1") -> StreamEvent:
    return StreamEvent(
        kind="tool_call", id=call_id, name=name, arguments_json=json.dumps(arguments)
    )


def _final_answer_script(text: str, finish_reason="stop") -> list[StreamEvent]:
    return [_token(text), _done(finish_reason=finish_reason)]


def _tool_call_script(name: str, arguments: dict, call_id: str = "call_1") -> list[StreamEvent]:
    return [_tool_call(name, arguments, call_id=call_id), _done(finish_reason="tool_calls")]


@dataclass
class _UserInput:
    kind: str  # "text" or "action"
    text: str | None = None
    action: str | None = None


def _mk_session():
    class _Session:
        subject_hint = None
        history = []

    return _Session()


def _budget():
    from tutor.app.prompt import Budget

    return Budget()


# ---------------------------------------------------------------------------
# Lesson 1: free-text question -> host pre-retrieves before the model call
# ---------------------------------------------------------------------------


def test_lesson1_pre_retrieve_happens_before_model_is_called():
    llm = FakeLlmClient([_final_answer_script("Water boils at 100C [S1].")])
    research = FakeResearchEngine()
    calc = FakeCalc()
    events = []

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="At what temperature does water boil?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=events.append,
    )

    assert research.call_count == 1
    assert llm.call_count == 1
    assert isinstance(result, TurnResult)
    assert result.route == "preretrieve"
    assert result.research_calls == 1
    assert "[S1]" in result.answer_text
    assert result.status == "ok"


def test_lesson1_pre_retrieve_emits_a_tool_event_before_the_first_model_call():
    """Regression test: the host's pre-retrieval (which happens for every
    free-text question, before the model is ever called) must be visible
    to the UI as a tool event, the same as a model-requested research
    follow-up is (see the `emit({"kind": "tool_result", "name":
    "research"})` call in the follow-up branch). Before this fix,
    run_turn never called `emit` for the pre-retrieval call at all, so a
    live turn's SSE stream carried no tool event ahead of its first
    token."""
    llm = FakeLlmClient([_final_answer_script("Water boils at 100C [S1].")])
    research = FakeResearchEngine()
    calc = FakeCalc()
    events: list = []

    run_turn(
        _mk_session(),
        _UserInput(kind="text", text="At what temperature does water boil?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=events.append,
    )

    kinds = [
        (e.get("kind"), e.get("name")) if isinstance(e, dict) else (e.kind, None)
        for e in events
    ]
    assert ("tool_result", "research") in kinds


def test_lesson1_evidence_appended_before_first_model_call():
    llm = FakeLlmClient([_final_answer_script("Answer [S1].")])
    research = FakeResearchEngine()
    calc = FakeCalc()

    run_turn(
        _mk_session(),
        _UserInput(kind="text", text="What is gravity?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    # The first (and only) model call's message log must already contain
    # the pre-retrieved evidence somewhere.
    first_call_messages = llm.calls[0]
    joined = json.dumps(first_call_messages)
    assert "Some evidence text." in joined or "evidence" in joined.lower()


# ---------------------------------------------------------------------------
# Lesson 2: one research follow-up (call 2 of 2)
# ---------------------------------------------------------------------------


def test_lesson2_single_followup_research_is_call_two_of_two():
    llm = FakeLlmClient(
        [
            _tool_call_script("research", {"query": "more detail on boiling point"}),
            _final_answer_script("Water boils at 100C at sea level [S1, S2]."),
        ]
    )
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="Why does water boil at 100C?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert research.call_count == 2
    assert llm.call_count == 2
    assert result.route == "preretrieve+followup"
    assert result.research_calls == 2
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# Lesson 3: a third research request hits the cap
# ---------------------------------------------------------------------------


def test_lesson3_third_research_request_is_cap_reached_not_executed():
    llm = FakeLlmClient(
        [
            _tool_call_script("research", {"query": "follow up one"}),
            _tool_call_script("research", {"query": "follow up two, over cap"}),
            _final_answer_script("Here is what I can tell you so far [S1]."),
        ]
    )
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="Explain in great depth why the sky is blue."),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    # Pre-retrieve (1) + one follow-up (2) = cap; the second follow-up must
    # NOT reach the real research engine a third time.
    assert research.call_count == 2
    assert llm.call_count == 3
    assert result.status == "ok"
    assert "sky is blue" in result.answer_text.lower() or result.answer_text

    # The third model call's own input must contain a cap-reached tool
    # result rather than fresh evidence.
    third_call_messages = llm.calls[2]
    joined = json.dumps(third_call_messages).lower()
    assert "cap" in joined


# ---------------------------------------------------------------------------
# Lesson 4: deterministic action -> no retrieval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["simpler", "deeper", "hint", "research_this"])
def test_lesson4_action_never_retrieves_and_routes_as_action(action):
    llm = FakeLlmClient([_final_answer_script("Here is a simpler explanation.")])
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="action", action=action),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    if action == "research_this":
        # research_this is explicitly an explicit-lookup action per spec
        # §12 ("same path as the default"); everything else never
        # retrieves.
        return

    assert research.call_count == 0
    assert result.route == "action"
    assert result.status == "ok"


def test_lesson4_simpler_action_route_is_exactly_action():
    llm = FakeLlmClient([_final_answer_script("Simpler explanation here.")])
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="action", action="simpler"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert research.call_count == 0
    assert result.route == "action"
    assert result.research_calls == 0


# ---------------------------------------------------------------------------
# Lesson 5: arithmetic -> calc tool, cap 4
# ---------------------------------------------------------------------------


def test_lesson5_calc_tool_called_and_result_in_answer():
    llm = FakeLlmClient(
        [
            _tool_call_script("calc", {"expression": "2 + 2"}),
            _final_answer_script("2 + 2 is 4."),
        ]
    )
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="What is 2 + 2?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert calc.call_count == 1
    assert result.calc_calls == 1
    assert "4" in result.answer_text
    assert result.status == "ok"


def test_assistant_tool_call_message_uses_openai_wire_format():
    """Regression test: llama-server's OpenAI-compatible endpoint rejects
    a `tool_calls` entry shaped `{"id", "name", "arguments_json"}` with
    HTTP 500 ("Missing tool call type") -- confirmed live against
    /v1/chat/completions. The assistant tool-call message run_turn
    appends to `messages` (replayed verbatim to llm.stream_chat on the
    next turn of the loop) must instead use the standard
    `{"id", "type": "function", "function": {"name", "arguments"}}`
    shape."""
    llm = FakeLlmClient(
        [
            _tool_call_script("calc", {"expression": "2 + 2"}),
            _final_answer_script("2 + 2 is 4."),
        ]
    )
    research = FakeResearchEngine()
    calc = FakeCalc()

    run_turn(
        _mk_session(),
        _UserInput(kind="text", text="What is 2 + 2?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    # llm.calls[1] is the messages list passed to the SECOND stream_chat
    # call, i.e. after the assistant's tool-call turn was appended.
    second_call_messages = llm.calls[1]
    assistant_msg = next(m for m in second_call_messages if m.get("role") == "assistant")
    tool_call_entry = assistant_msg["tool_calls"][0]

    assert tool_call_entry.get("type") == "function"
    assert "id" in tool_call_entry
    assert "function" in tool_call_entry
    assert tool_call_entry["function"]["name"] == "calc"
    assert "arguments" in tool_call_entry["function"]
    assert "name" not in tool_call_entry
    assert "arguments_json" not in tool_call_entry


def test_lesson5_calc_calls_do_not_count_toward_research_cap():
    llm = FakeLlmClient(
        [
            _tool_call_script("calc", {"expression": "1 + 1"}),
            _tool_call_script("research", {"query": "one more fact"}),
            _final_answer_script("Done [S1, S2]."),
        ]
    )
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="Calculate 1+1 and tell me a related fact."),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert calc.call_count == 1
    # pre-retrieve (1) + the one research follow-up (2) still fits the cap.
    assert research.call_count == 2
    assert result.status == "ok"


def test_lesson5_fifth_calc_call_in_same_turn_hits_cap():
    scripts = [_tool_call_script("calc", {"expression": str(i)}) for i in range(5)]
    scripts.append(_final_answer_script("Done computing."))
    llm = FakeLlmClient(scripts)
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="Do five calculations for me."),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    # Only 4 of the 5 requested calc calls reach the real sandboxed calc.
    assert calc.call_count == 4
    assert result.calc_calls == 4
    assert result.status == "ok"

    last_tool_call_messages = llm.calls[-1]
    joined = json.dumps(last_tool_call_messages).lower()
    assert "cap" in joined


# ---------------------------------------------------------------------------
# Lesson 6: malformed / invalid tool calls, and prose that looks like one
# ---------------------------------------------------------------------------


def test_lesson6_invalid_json_arguments_not_executed_validation_error_returned():
    bad_call = StreamEvent(
        kind="tool_call", id="call_1", name="calc", arguments_json="{not valid json"
    )
    llm = FakeLlmClient(
        [
            [bad_call, _done(finish_reason="tool_calls")],
            _final_answer_script("Sorry, let me try a valid calculation instead."),
        ]
    )
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="What is 2 + 2?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert calc.call_count == 0
    assert result.status == "ok"
    second_call_messages = llm.calls[1]
    joined = json.dumps(second_call_messages).lower()
    assert "invalid" in joined or "error" in joined


def test_lesson6_unknown_tool_name_not_executed_validation_error_returned():
    bad_call = StreamEvent(
        kind="tool_call", id="call_1", name="delete_everything", arguments_json="{}"
    )
    llm = FakeLlmClient(
        [
            [bad_call, _done(finish_reason="tool_calls")],
            _final_answer_script("I can't do that, here's an answer instead."),
        ]
    )
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="Delete everything on my computer."),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert calc.call_count == 0
    assert research.call_count == 1  # only the pre-retrieve; tool itself never ran
    assert result.status == "ok"


def test_lesson6_schema_violation_extra_field_not_executed():
    bad_call = StreamEvent(
        kind="tool_call",
        id="call_1",
        name="calc",
        arguments_json=json.dumps({"expression": "2+2", "unexpected_field": True}),
    )
    llm = FakeLlmClient(
        [
            [bad_call, _done(finish_reason="tool_calls")],
            _final_answer_script("Here's the answer anyway."),
        ]
    )
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="What is 2+2?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert calc.call_count == 0
    assert result.status == "ok"


def test_lesson6_prose_that_looks_like_a_tool_call_is_never_parsed_or_executed():
    # The model's plain-text answer literally contains something that
    # *looks* like a tool call, but arrives as a "token" event, never a
    # "tool_call" event -- the host must not regex/parse this out of prose.
    lookalike_text = 'Sure: {"name": "calc", "arguments": {"expression": "2+2"}}'
    llm = FakeLlmClient([_final_answer_script(lookalike_text)])
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="Say something that looks like a tool call."),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert calc.call_count == 0
    assert research.call_count == 1  # only the ordinary pre-retrieve
    assert result.status == "ok"
    assert lookalike_text in result.answer_text


# ---------------------------------------------------------------------------
# Cross-cutting: llm error event, cancellation, emitted events, logging fields
# ---------------------------------------------------------------------------


def test_llm_error_event_yields_error_status_no_exception():
    llm = FakeLlmClient([[StreamEvent(kind="error", message="connection refused")]])
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="What is the capital of France?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert result.status == "error"
    assert isinstance(result.answer_text, str)
    assert result.answer_text  # student-safe message, non-empty


def test_cancel_event_yields_cancelled_status():
    cancel = threading.Event()
    cancel.set()
    llm = FakeLlmClient([[_done(finish_reason="cancelled")]])
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="Tell me about volcanoes."),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
        cancel=cancel,
    )

    assert result.status == "cancelled"


def test_emitted_events_include_token_deltas_tool_activity_and_citations():
    llm = FakeLlmClient(
        [
            _tool_call_script("calc", {"expression": "2 + 2"}),
            _final_answer_script("2 + 2 is 4."),
        ]
    )
    research = FakeResearchEngine()
    calc = FakeCalc()
    events = []

    run_turn(
        _mk_session(),
        _UserInput(kind="text", text="What is 2 + 2?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=events.append,
    )

    kinds = [
        getattr(e, "kind", None) or (e.get("kind") if isinstance(e, dict) else None)
        for e in events
    ]
    assert any(k == "token" for k in kinds)
    assert any(k in ("tool_call", "tool_result", "tool_activity") for k in kinds)


def test_turn_result_logs_route_calc_calls_research_calls_and_cached_tokens():
    llm = FakeLlmClient([_final_answer_script("Answer [S1].", finish_reason="stop")])
    research = FakeResearchEngine()
    calc = FakeCalc()

    result = run_turn(
        _mk_session(),
        _UserInput(kind="text", text="What is photosynthesis?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert hasattr(result, "route")
    assert hasattr(result, "calc_calls")
    assert hasattr(result, "research_calls")
    # cached tokens is reported when the llm reports it; default 0/None is fine
    assert hasattr(result, "cached_tokens")


def test_researched_passages_are_retained_on_the_session_for_citation_resolution():
    """Regression test: run_turn must hand every research packet's passages
    to session.retain_passages(...) so citations (resolved from
    session.known_passages() by the app layer, e.g.
    tutor.app.compose._make_turn_runner) can ever resolve a [S#] label.
    Before this fix, run_turn never called retain_passages, so
    session.known_passages() was always empty and every citation in a live
    answer would resolve as unresolved."""

    class _RetainingSession:
        subject_hint = None
        history: list = []

        def __init__(self):
            self.retained: list[dict] = []

        def retain_passages(self, passages):
            self.retained.extend(passages)

    llm = FakeLlmClient([_final_answer_script("Water boils at 100C [S1].")])
    research = FakeResearchEngine()
    calc = FakeCalc()
    session = _RetainingSession()

    run_turn(
        session,
        _UserInput(kind="text", text="What is the boiling point of water?"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=_budget(),
        emit=lambda e: None,
    )

    assert session.retained, "expected the pre-retrieved passages to be retained"
    assert any(p.get("label") == "S1" for p in session.retained)

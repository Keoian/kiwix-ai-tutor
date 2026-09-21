"""Tests for ``app.no_specifics_without_source`` and
``app.child_safe_body_topics`` (docs/rewrite_on_weak_evidence.md, "No
specifics without a source" / "Questions about bodies, sex and growing
up"): a strong system-prompt instruction (with exemplars) against
inventing proper names/exact numbers when the library sources don't
supply them, mutually-exclusive evidence-tail wording by evidence level,
and a system-prompt section keeping body/sex/growing-up answers clinical
and library-only. All fakes; no live LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tutor.app.agent_loop import (
    _CHILD_SAFE_BODY_TOPICS_SECTION,
    _NAMES_NUMBERS_SECTION,
    _not_found_tool_text,
    run_turn,
)
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
        script = self._scripts.pop(0)
        yield from script


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
        self.calls.append({"query": query})
        response = self._responses.pop(0)
        response.assessment = assess_evidence(query, response)
        return response


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
    session = Session(count_tokens=_count_tokens, subject_hint="general")
    return session, Budget()


_EMPTY_RESPONSE = _Response(passages=[])


# --------------------------------------------------------------------
# _not_found_tool_text: exact tail text per evidence level
# --------------------------------------------------------------------


def test_not_found_text_weak_appends_one_line_to_old_text():
    old_text = _not_found_tool_text(
        searched_for=[], level_after="weak", no_specifics_without_source=False
    )
    new_text = _not_found_tool_text(
        searched_for=[], level_after="weak", no_specifics_without_source=True
    )
    assert new_text.startswith(old_text)
    assert new_text != old_text
    assert (
        "These sources may not answer the question. Use only what they "
        "actually say; give no names or numbers from memory."
    ) in new_text


def test_not_found_text_empty_is_fully_replaced():
    text = _not_found_tool_text(
        searched_for=[], level_after="empty", no_specifics_without_source=True
    )
    assert text == (
        "The library search found nothing for this. Do not give proper "
        "names, exact numbers, dates or records from memory. Explain the "
        "general idea if you can, say what you couldn't find, and suggest "
        "one thing to look up next."
    )
    assert "No good match was found" not in text


def test_not_found_text_off_is_old_bytes_for_any_level():
    weak = _not_found_tool_text(
        searched_for=[], level_after="weak", no_specifics_without_source=False
    )
    empty = _not_found_tool_text(
        searched_for=[], level_after="empty", no_specifics_without_source=False
    )
    assert weak.startswith("No good match was found in the library")
    assert empty.startswith("No good match was found in the library")
    assert "weak evidence" in weak
    assert "empty evidence" in empty


# --------------------------------------------------------------------
# System prompt section presence/absence
# --------------------------------------------------------------------


def _system_text_from(session, budget, llm, research, *, no_specifics, child_safe):
    calc = FakeCalc()
    run_turn(
        session,
        _UserInput(kind="text", text="tell me about xyzzy"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: None,
        model_writes_search=False,
        no_specifics_without_source=no_specifics,
        child_safe_body_topics=child_safe,
    )
    rendered = session.log.render()
    return rendered[0]["content"]


def test_names_numbers_section_present_by_default():
    session, budget = _mk_session()
    llm = FakeLlmClient([_tool_call("research", {"queries": ["xyzzy"]}), _final("ok [S1].")])
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _EMPTY_RESPONSE])
    system_text = _system_text_from(
        session, budget, llm, research, no_specifics=True, child_safe=False
    )
    assert _NAMES_NUMBERS_SECTION in system_text
    assert _CHILD_SAFE_BODY_TOPICS_SECTION not in system_text


def test_names_numbers_section_absent_when_off():
    session, budget = _mk_session()
    llm = FakeLlmClient([_tool_call("research", {"queries": ["xyzzy"]}), _final("ok [S1].")])
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _EMPTY_RESPONSE])
    system_text = _system_text_from(
        session, budget, llm, research, no_specifics=False, child_safe=False
    )
    assert _NAMES_NUMBERS_SECTION not in system_text


def test_child_safe_body_topics_section_present_by_default():
    session, budget = _mk_session()
    llm = FakeLlmClient([_tool_call("research", {"queries": ["xyzzy"]}), _final("ok [S1].")])
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _EMPTY_RESPONSE])
    system_text = _system_text_from(
        session, budget, llm, research, no_specifics=False, child_safe=True
    )
    assert _CHILD_SAFE_BODY_TOPICS_SECTION in system_text


def test_child_safe_body_topics_section_absent_when_off():
    session, budget = _mk_session()
    llm = FakeLlmClient([_tool_call("research", {"queries": ["xyzzy"]}), _final("ok [S1].")])
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _EMPTY_RESPONSE])
    system_text = _system_text_from(
        session, budget, llm, research, no_specifics=False, child_safe=False
    )
    assert _CHILD_SAFE_BODY_TOPICS_SECTION not in system_text


def test_both_settings_off_reproduces_old_system_prompt_bytes():
    from tutor.app.agent_loop import _load_default_system_text

    session, budget = _mk_session()
    llm = FakeLlmClient([_tool_call("research", {"queries": ["xyzzy"]}), _final("ok [S1].")])
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _EMPTY_RESPONSE])
    system_text = _system_text_from(
        session, budget, llm, research, no_specifics=False, child_safe=False
    )
    assert system_text == _load_default_system_text()


# --------------------------------------------------------------------
# Status string
# --------------------------------------------------------------------


def test_not_found_status_text_when_on():
    session, budget = _mk_session()
    llm = FakeLlmClient([_tool_call("research", {"queries": ["xyzzy"]}), _final("ok.")])
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _EMPTY_RESPONSE])
    calc = FakeCalc()
    events = []
    run_turn(
        session,
        _UserInput(kind="text", text="tell me about xyzzy"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: events.append(e),
        model_writes_search=False,
        no_specifics_without_source=True,
    )
    detail = next(e["detail"] for e in events if e.get("stage") == "not_found")
    assert detail == "Nothing in the library on this. Answering carefully..."


def test_not_found_status_text_when_off_is_old_bytes():
    session, budget = _mk_session()
    llm = FakeLlmClient([_tool_call("research", {"queries": ["xyzzy"]}), _final("ok.")])
    research = ScriptedResearchEngine([_EMPTY_RESPONSE, _EMPTY_RESPONSE])
    calc = FakeCalc()
    events = []
    run_turn(
        session,
        _UserInput(kind="text", text="tell me about xyzzy"),
        llm=llm,
        research_engine=research,
        calc=calc,
        budget=budget,
        emit=lambda e: events.append(e),
        model_writes_search=False,
        no_specifics_without_source=False,
    )
    detail = next(e["detail"] for e in events if e.get("stage") == "not_found")
    assert detail == "Nothing in the library on this. Answering from what I know..."

"""The tutor's per-turn agent loop: receive -> route -> pre-retrieve ->
build prompt -> inference -> tool dispatch -> inference -> finish.

Authoritative sources: docs/plan/offline_tutor_spec_v0.3.md §6 (turn flow;
the research cap message: "If the model requests a second `research`
after the cap, the host returns a tool result stating the cap is reached
and the model answers the supported portion or asks for clarification.")
and docs/plan/offline_tutor_implementation_plan.md WP-C2 (`research` cap 2
INCLUDING the host pre-retrieval, `calc` cap 4, schema validation, never
parse prose as a tool call).

The six scripted lessons (this module's own enumeration, see
tests/test_agent_loop.py for the authoritative list under test):

1. Free-text question: host pre-retrieves before the first model call.
2. The model's first reply is a `research` follow-up: executed as call 2
   of the research cap.
3. A third research request hits the cap: the host returns a cap-reached
   tool result instead of calling the real research engine again.
4. A deterministic action never retrieves.
5. Arithmetic free text: the model calls `calc` (cap 4 per turn, separate
   from the research cap).
6. A malformed tool call (bad JSON, unknown name, schema violation) is
   validated and, on failure, never executed -- the host returns a
   validation-error tool result instead. Prose that merely looks like a
   tool call is never parsed as one: only a "tool_call" StreamEvent from
   the LLM client is ever treated as a tool call.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

from tutor.app.citations import extract_labels, render_evidence
from tutor.app.prompt import PromptOverflow
from tutor.tools.schemas import TOOLS, validate_tool_call

RESEARCH_CAP = 2
CALC_CAP = 4

_STUDENT_SAFE_ERROR = (
    "Sorry, I ran into a problem answering that. Please try asking again."
)


@dataclass
class TurnResult:
    status: str  # "ok" | "error" | "cancelled"
    answer_text: str
    route: str
    research_calls: int = 0
    calc_calls: int = 0
    cached_tokens: int | None = 0
    events: list = field(default_factory=list)


def _passage_to_dict(passage) -> dict:
    if isinstance(passage, dict):
        return dict(passage)
    return {
        "id": getattr(passage, "passage_id", None) or getattr(passage, "id", None),
        "label": getattr(passage, "label", None),
        "title": getattr(passage, "title", None),
        "path": getattr(passage, "path", None),
        "text": getattr(passage, "text", ""),
        "kind": getattr(passage, "kind", "article"),
    }


def _retain_passages(session, packet: dict) -> None:
    """Hand a research packet's passages to ``session.retain_passages``
    (if the session supports it) so citations can be resolved later from
    ``session.known_passages()`` -- including after prompt-log eviction.
    Sessions without the method (some fakes in unit tests) are a no-op."""
    retain = getattr(session, "retain_passages", None)
    if retain is not None:
        retain(packet.get("passages", []))


def _packet_from_response(response) -> dict:
    passages = getattr(response, "passages", None)
    if passages is None and isinstance(response, dict):
        passages = response.get("passages", [])
    return {"passages": [_passage_to_dict(p) for p in (passages or [])]}


def _to_wire_messages(rendered: list[dict]) -> list[dict]:
    """Translate ``PromptLog.render()``'s internal message shapes into the
    OpenAI-compatible wire shapes llama-server's ``/v1/chat/completions``
    accepts: evidence's ``{"role": "tool", "passages": [...]}`` becomes a
    plain ``{"role": "tool", "content": <rendered evidence text>}``, and
    an assistant entry's internal ``cited_labels`` bookkeeping field
    (never part of the wire schema) is dropped."""
    wire: list[dict] = []
    for message in rendered:
        if message.get("passages") is not None:
            wire.append(
                {"role": message["role"], "content": render_evidence(message)}
            )
        elif message.get("role") == "assistant" and "cited_labels" in message:
            wire.append({"role": "assistant", "content": message.get("content")})
        else:
            wire.append(message)
    return wire


def _trim_and_append_evidence(log, passages: list[dict], budget) -> None:
    """Append ``passages`` as an evidence packet, trimming the
    lowest-ranked (last) passages first if the full packet does not fit
    ``budget.newest`` -- never crashing the turn on PromptOverflow."""
    remaining = list(passages)
    while True:
        try:
            log.append_evidence(remaining, budget=budget)
            return
        except PromptOverflow:
            if not remaining:
                raise
            remaining = remaining[:-1]


def run_turn(
    session,
    user_input,
    *,
    llm,
    research_engine,
    calc,
    budget,
    emit,
    cancel: threading.Event | None = None,
) -> TurnResult:
    research_calls = 0
    calc_calls = 0
    followup_research_used = False
    events: list = []

    log = getattr(session, "log", None)
    use_log = log is not None and hasattr(log, "append_user")

    system_text = (
        "You are an offline tutor. Teach rather than answer outright, cite "
        "[S#] for source-backed statements, and use the calc tool before "
        "asserting arithmetic beyond single-digit numbers."
    )
    profile_summary = getattr(session, "profile_summary", None)
    if profile_summary:
        system_text = f"{system_text} {profile_summary}"

    if use_log:
        try:
            log.append_system(system_text)
        except ValueError:
            pass  # already appended on a prior turn of this session
    else:
        messages: list[dict] = [{"role": "system", "content": system_text}]

    if user_input.kind == "action":
        route = "action"
        if use_log:
            log.append_user(f"[action:{user_input.action}]")
        else:
            messages.append({"role": "user", "content": f"[action:{user_input.action}]"})
    else:
        route = "preretrieve"
        if use_log:
            log.append_user(user_input.text)
        else:
            messages.append({"role": "user", "content": user_input.text})
        response = research_engine.research(
            user_input.text, topic_hint=getattr(session, "subject_hint", None)
        )
        research_calls += 1
        packet = _packet_from_response(response)
        _retain_passages(session, packet)
        if use_log:
            _trim_and_append_evidence(log, packet["passages"], budget)
        else:
            evidence_text = render_evidence(packet)
            messages.append({"role": "tool", "content": f"[research results]\n{evidence_text}"})
        emit({"kind": "tool_result", "name": "research"})

    while True:
        if cancel is not None and cancel.is_set():
            return TurnResult(
                status="cancelled",
                answer_text="",
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                events=events,
            )

        if use_log:
            eviction_event = log.evict(budget)
            if eviction_event is not None:
                events.append(eviction_event)
                emit(
                    {
                        "kind": "eviction_reprefill",
                        "evicted_turns": eviction_event.evicted_turns,
                        "tokens_before": eviction_event.tokens_before,
                        "tokens_after": eviction_event.tokens_after,
                    }
                )
            messages = _to_wire_messages(log.render())

        answer_text_parts: list[str] = []
        tool_calls: list = []
        finish_reason = "stop"
        usage: dict = {}
        errored = False

        for evt in llm.stream_chat(messages, tools=TOOLS, cancel=cancel):
            if evt.kind == "token":
                answer_text_parts.append(evt.text or "")
                emit(evt)
            elif evt.kind == "tool_call":
                tool_calls.append(evt)
                emit(evt)
            elif evt.kind == "error":
                errored = True
                emit(evt)
                break
            elif evt.kind == "done":
                finish_reason = evt.finish_reason or "stop"
                usage = evt.usage or {}

        if errored:
            return TurnResult(
                status="error",
                answer_text=_STUDENT_SAFE_ERROR,
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                events=events,
            )

        if finish_reason == "cancelled":
            return TurnResult(
                status="cancelled",
                answer_text="".join(answer_text_parts),
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                events=events,
            )

        if not tool_calls:
            answer_text = "".join(answer_text_parts)
            if use_log:
                log.append_assistant(answer_text, cited_labels=extract_labels(answer_text))
            return TurnResult(
                status="ok",
                answer_text=answer_text,
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                cached_tokens=usage.get("cached_tokens", 0),
                events=events,
            )

        # Record the assistant's tool-call turn, then dispatch each call.
        wire_tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments_json,
                },
            }
            for tc in tool_calls
        ]
        if use_log:
            log.append_assistant_tool_calls(wire_tool_calls)
        else:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": wire_tool_calls}
            )

        def _append_tool_message(tool_call_id: str, content: str, _messages=messages) -> None:
            if use_log:
                log.append_tool_result(tool_call_id=tool_call_id, content=content)
            else:
                _messages.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": content}
                )

        for tc in tool_calls:
            validation = validate_tool_call(tc.name, tc.arguments_json or "{}")
            if not validation.ok:
                _append_tool_message(tc.id, f"error: invalid tool call: {validation.error}")
                emit({"kind": "tool_result", "name": tc.name, "ok": False})
                continue

            if tc.name == "research":
                if research_calls >= RESEARCH_CAP:
                    _append_tool_message(
                        tc.id,
                        f"research cap of {RESEARCH_CAP} calls reached for this "
                        "turn; answer with the supported portion or ask for "
                        "clarification.",
                    )
                else:
                    response = research_engine.research(
                        validation.arguments["query"],
                        topic_hint=getattr(session, "subject_hint", None),
                        keywords=validation.arguments.get("keywords"),
                    )
                    research_calls += 1
                    followup_research_used = True
                    packet = _packet_from_response(response)
                    _retain_passages(session, packet)
                    if use_log:
                        _trim_and_append_evidence(log, packet["passages"], budget)
                    else:
                        evidence_text = render_evidence(packet)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"[research results]\n{evidence_text}",
                            }
                        )
                emit({"kind": "tool_result", "name": "research"})
            elif tc.name == "calc":
                if calc_calls >= CALC_CAP:
                    _append_tool_message(
                        tc.id, f"calc cap of {CALC_CAP} calls reached for this turn."
                    )
                else:
                    calc_result = calc.evaluate(validation.arguments["expression"])
                    calc_calls += 1
                    _append_tool_message(tc.id, json.dumps(calc_result))
                emit({"kind": "tool_result", "name": "calc"})
            else:
                # Unreachable: validate_tool_call already rejects unknown
                # tool names, but keep the loop total in case of future
                # tool additions with no dispatch branch yet.
                _append_tool_message(tc.id, f"error: no dispatcher for tool {tc.name}")

        if followup_research_used and route == "preretrieve":
            route = "preretrieve+followup"

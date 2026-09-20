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

from tutor.app.citations import render_evidence
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
    del budget  # reserved for prompt-log integration; not needed by these unit tests

    research_calls = 0
    calc_calls = 0
    followup_research_used = False

    system_text = (
        "You are an offline tutor. Teach rather than answer outright, cite "
        "[S#] for source-backed statements, and use the calc tool before "
        "asserting arithmetic beyond single-digit numbers."
    )
    messages: list[dict] = [{"role": "system", "content": system_text}]

    if user_input.kind == "action":
        route = "action"
        messages.append({"role": "user", "content": f"[action:{user_input.action}]"})
    else:
        route = "preretrieve"
        messages.append({"role": "user", "content": user_input.text})
        response = research_engine.research(
            user_input.text, topic_hint=getattr(session, "subject_hint", None)
        )
        research_calls += 1
        packet = _packet_from_response(response)
        _retain_passages(session, packet)
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
            )

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
            )

        if finish_reason == "cancelled":
            return TurnResult(
                status="cancelled",
                answer_text="".join(answer_text_parts),
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
            )

        if not tool_calls:
            answer_text = "".join(answer_text_parts)
            return TurnResult(
                status="ok",
                answer_text=answer_text,
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                cached_tokens=usage.get("cached_tokens", 0),
            )

        # Record the assistant's tool-call turn, then dispatch each call.
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments_json,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            validation = validate_tool_call(tc.name, tc.arguments_json or "{}")
            if not validation.ok:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"error: invalid tool call: {validation.error}",
                    }
                )
                emit({"kind": "tool_result", "name": tc.name, "ok": False})
                continue

            if tc.name == "research":
                if research_calls >= RESEARCH_CAP:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                f"research cap of {RESEARCH_CAP} calls reached for this "
                                "turn; answer with the supported portion or ask for "
                                "clarification."
                            ),
                        }
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
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                f"calc cap of {CALC_CAP} calls reached for this turn."
                            ),
                        }
                    )
                else:
                    calc_result = calc.evaluate(validation.arguments["expression"])
                    calc_calls += 1
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(calc_result),
                        }
                    )
                emit({"kind": "tool_result", "name": "calc"})
            else:
                # Unreachable: validate_tool_call already rejects unknown
                # tool names, but keep the loop total in case of future
                # tool additions with no dispatch branch yet.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"error: no dispatcher for tool {tc.name}",
                    }
                )

        if followup_research_used and route == "preretrieve":
            route = "preretrieve+followup"

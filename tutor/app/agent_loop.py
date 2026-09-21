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
from pathlib import Path

from tutor.app.citations import extract_labels, render_evidence
from tutor.app.prompt import PromptOverflow
from tutor.app.repetition_guard import find_repetition_loop
from tutor.tools.schemas import TOOLS, validate_tool_call

RESEARCH_CAP = 2
CALC_CAP = 4

_STUDENT_SAFE_ERROR = (
    "Sorry, I ran into a problem answering that. Please try asking again."
)

_SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")


class _UnionCancel:
    """Duck-typed ``threading.Event``-alike (only ``is_set()`` is used by
    ``LlamaClient.stream_chat``) that is set when ANY of the given events
    is set. Lets the repetition-loop guard (below) trigger the exact same
    clean-stream-close path as a real user cancel -- closing the HTTP
    connection to llama-server properly, never killing the server --
    while still letting ``run_turn`` tell the two apart afterwards by
    checking the original ``cancel`` event directly."""

    def __init__(self, *events) -> None:
        self._events = [e for e in events if e is not None]

    def is_set(self) -> bool:
        return any(e.is_set() for e in self._events)


def _load_default_system_text() -> str:
    """Read the host's default system prompt from ``system_prompt.txt``
    (kept in a plain-text file so it's easy to review/diff independently
    of the code, and so ``eval/run_turn_eval.py`` can swap variants in
    without touching this module's source)."""
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


@dataclass
class TurnResult:
    status: str  # "ok" | "error" | "cancelled"
    answer_text: str
    route: str
    research_calls: int = 0
    calc_calls: int = 0
    cached_tokens: int | None = 0
    prompt_tokens: int | None = None
    events: list = field(default_factory=list)
    uncited: bool = False
    """True when the route was a factual pre-retrieval (evidence was
    supplied) but the model's final answer contains no [S#] label at all.
    Per spec §11/§12 the host never fabricates a citation the model did
    not write; this flag only tells the UI to show an "uncited" notice
    (§12's chat pane already distinguishes source-backed/computed/own-
    example statement styles, so this is the same kind of provenance
    signal, not a new citation)."""
    truncated: str | None = None
    """``None`` (not truncated), ``"repetition"`` (the host-side
    repetition-loop guard in tutor.app.repetition_guard stopped
    generation and trimmed the answer), or ``"max_tokens"`` (the server's
    own ``finish_reason == "length"``, i.e. the ``max_tokens`` cap was
    hit). Citations and attributions still run on the (possibly trimmed)
    ``answer_text``; the host never fabricates content to fill in what
    was cut."""


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
    system_text_override: str | None = None,
    temperature: float | None = None,
) -> TurnResult:
    research_calls = 0
    calc_calls = 0
    followup_research_used = False
    events: list = []

    log = getattr(session, "log", None)
    use_log = log is not None and hasattr(log, "append_user")

    system_text = (
        system_text_override
        if system_text_override is not None
        else _load_default_system_text()
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
        emit({"kind": "status", "stage": "searching", "detail": "Looking in the library..."})
        response = research_engine.research(
            user_input.text, topic_hint=getattr(session, "subject_hint", None)
        )
        research_calls += 1
        packet = _packet_from_response(response)
        _retain_passages(session, packet)
        emit(
            {
                "kind": "status",
                "stage": "reading",
                "detail": f"Reading {len(packet.get('passages') or [])} sources...",
            }
        )
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
                        "dropped_uncited_passages": eviction_event.dropped_uncited_passages,
                        "dropped_uncited_tokens": eviction_event.dropped_uncited_tokens,
                    }
                )
            messages = _to_wire_messages(log.render())

        answer_text_parts: list[str] = []
        tool_calls: list = []
        finish_reason = "stop"
        usage: dict = {}
        errored = False
        truncated: str | None = None
        trimmed_answer_text: str | None = None
        # The guard's own cancel signal, ORed with the caller's `cancel`
        # (if any) so LlamaClient.stream_chat closes the connection the
        # same clean way a user-initiated stop does (see _UnionCancel).
        guard_cancel = threading.Event()
        stream_cancel = _UnionCancel(cancel, guard_cancel)

        emit({"kind": "status", "stage": "thinking", "detail": "Writing an answer..."})
        for evt in llm.stream_chat(
            messages,
            tools=TOOLS,
            cancel=stream_cancel,
            temperature=temperature,
            max_tokens=budget.generation,
        ):
            if evt.kind == "token":
                answer_text_parts.append(evt.text or "")
                emit(evt)
                if truncated is None:
                    loop = find_repetition_loop("".join(answer_text_parts))
                    if loop is not None:
                        truncated = "repetition"
                        trimmed_answer_text = loop.trimmed_text
                        guard_cancel.set()
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
                if finish_reason == "length" and truncated is None:
                    truncated = "max_tokens"

        if errored:
            return TurnResult(
                status="error",
                answer_text=_STUDENT_SAFE_ERROR,
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                events=events,
            )

        # A "cancelled" finish_reason from the guard's own union event
        # (not the caller's `cancel`) is not a real user cancellation --
        # it's the repetition guard stopping generation cleanly. Only
        # treat it as a cancelled turn when the caller actually asked to
        # cancel.
        if finish_reason == "cancelled" and (cancel is None or not cancel.is_set()):
            finish_reason = "stop"

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
            answer_text = (
                trimmed_answer_text if truncated == "repetition" else "".join(answer_text_parts)
            )
            labels = extract_labels(answer_text)
            if use_log:
                log.append_assistant(answer_text, cited_labels=labels)
            uncited = route.startswith("preretrieve") and not labels
            return TurnResult(
                status="ok",
                answer_text=answer_text,
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                cached_tokens=usage.get("cached_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens"),
                events=events,
                uncited=uncited,
                truncated=truncated,
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

        turn_cancelled = False
        for index, tc in enumerate(tool_calls):
            # Review pass 2, finding 1: once the assistant's tool_calls
            # message has been appended (above), every id it names MUST get
            # a matching tool-result entry -- on a raised exception, on a
            # cancel mid-loop, or on normal dispatch -- or the log is left
            # with a dangling tool_calls message that the next turn cannot
            # replay (OpenAI-compatible endpoints reject it, wedging the
            # session/lesson permanently). So this loop body never lets an
            # exception propagate past a single tool call, and a cancel
            # mid-loop fills in the remaining not-yet-dispatched ids with a
            # synthetic "cancelled" result before returning, instead of
            # leaving them unfilled.
            if cancel is not None and cancel.is_set():
                turn_cancelled = True
                for remaining_tc in tool_calls[index:]:
                    _append_tool_message(
                        remaining_tc.id,
                        json.dumps({"ok": False, "error": "turn cancelled"}),
                    )
                    emit({"kind": "tool_result", "name": remaining_tc.name, "ok": False})
                break

            try:
                validation = validate_tool_call(tc.name, tc.arguments_json or "{}")
                if not validation.ok:
                    _append_tool_message(tc.id, f"error: invalid tool call: {validation.error}")
                    emit({"kind": "tool_result", "name": tc.name, "ok": False})
                    continue

                if tc.name == "research":
                    emit(
                        {
                            "kind": "status",
                            "stage": "tool",
                            "detail": "Searching again...",
                        }
                    )
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
                    emit(
                        {
                            "kind": "status",
                            "stage": "tool",
                            "detail": "Using the calculator...",
                        }
                    )
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
            except Exception as exc:  # noqa: BLE001 - never leave the log dangling
                # research/calc (or evidence trimming) raised -- e.g. a
                # ZimWorker timeout not swallowed at a lower layer. Give
                # this call's id a student-safe, model-safe error result
                # instead of letting the exception propagate: propagating
                # here would leave `log` with an assistant tool_calls
                # message and no result for this id, which is exactly the
                # corruption this fix exists to prevent. Every dispatch
                # branch above only appends its tool message *after*
                # successfully computing a result, so reaching this except
                # means no message for `tc.id` has been appended yet.
                _append_tool_message(
                    tc.id, json.dumps({"ok": False, "error": str(exc)})
                )
                emit({"kind": "tool_result", "name": tc.name, "ok": False})

        if use_log:
            log.validate()

        if turn_cancelled:
            return TurnResult(
                status="cancelled",
                answer_text="".join(answer_text_parts),
                route=route,
                research_calls=research_calls,
                calc_calls=calc_calls,
                events=events,
            )

        if followup_research_used and route == "preretrieve":
            route = "preretrieve+followup"

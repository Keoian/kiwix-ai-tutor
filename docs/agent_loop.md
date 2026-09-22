# Agent loop

`tutor.app.agent_loop.run_turn(session, user_input, *, llm, research_engine,
calc, budget, emit, cancel=None) -> TurnResult` implements one turn of the
spec's turn flow (docs/plan/offline_tutor_spec_v0.3.md §6):

receive -> route by action-vs-free-text -> pre-retrieve -> build prompt ->
inference request 1 -> tool dispatch -> inference request N -> finish.

## Routing

- `user_input.kind == "action"` (a deterministic UI action such as
  "simpler", "deeper", "hint") routes as `"action"` and never retrieves.
  `research_this` is the one action that is an explicit lookup per spec
  §12 and is handled the same way as free text elsewhere in the app, but
  is not itself part of this loop's routing distinction under test.
- `user_input.kind == "text"` (free-text question) routes as
  `"preretrieve"`: the host calls `research_engine.research(...)` once,
  *before* the model is ever called, and appends the result as a
  tool-result-shaped message. If the model's own first reply is itself a
  `research` follow-up call, the route becomes `"preretrieve+followup"`
  once that follow-up actually executes.

## Caps

- **research cap = 2 per turn**, counting the host's own pre-retrieve as
  call 1. A third request (pre-retrieve + one follow-up already used the
  cap) is never sent to the real research engine: the host returns a
  cap-reached tool result instead, and the model's next reply (over
  whatever evidence it already has) is what the turn returns.
- **calc cap = 4 per turn**, tracked independently of the research cap. A
  fifth calc call in the same turn gets a cap-reached tool result instead
  of being run.
- Both caps are enforced by `run_turn` itself, not by the tools: `calc`
  and the research engine are never called once a cap is hit.

## Tool-call validation

Every `tool_call` `StreamEvent` from the LLM client is validated with
`tutor.tools.schemas.validate_tool_call` before it is ever dispatched:

- invalid JSON arguments,
- an unknown tool name, or
- a schema violation (missing required field, unexpected extra field,
  wrong type)

each produce a validation-error tool-result message back to the model
*instead of* running the real tool, and the turn still completes
normally. Prose that merely looks like a tool call (e.g. literal
`{"name": "calc", ...}` inside the model's plain-text answer) is never
parsed as one: only an actual `"tool_call"` `StreamEvent` from the LLM
client (as opposed to a `"token"` event containing that text) is ever
treated as a tool call. The host does no regex/prose scanning for tool
calls at all.

## The six scripted lessons

These are the task brief's own enumeration (spec/plan text does not spell
out a numbered list of six by name); all six are exercised as pure unit
tests in `tests/test_agent_loop.py` against a scripted fake LLM client and
fake research/calc callables (no network, no real model, no real
subprocess sandbox):

1. **Free-text factual question** — host pre-retrieves before the model is
   called at all. `test_lesson1_pre_retrieve_happens_before_model_is_called`,
   `test_lesson1_evidence_appended_before_first_model_call`.
2. **One research follow-up** — the model's first reply is itself a
   `research` call, executed as call 2 of the cap.
   `test_lesson2_single_followup_research_is_call_two_of_two`.
3. **A third research request hits the cap** — not executed against the
   real engine; the model's final reply is still returned.
   `test_lesson3_third_research_request_is_cap_reached_not_executed`.
4. **A deterministic action never retrieves.**
   `test_lesson4_action_never_retrieves_and_routes_as_action`,
   `test_lesson4_simpler_action_route_is_exactly_action`.
5. **Arithmetic free text calls `calc`; cap 4.**
   `test_lesson5_calc_tool_called_and_result_in_answer`,
   `test_lesson5_calc_calls_do_not_count_toward_research_cap`,
   `test_lesson5_fifth_calc_call_in_same_turn_hits_cap`.
6. **Malformed/invalid tool calls are validated, not executed; prose that
   looks like a tool call is never parsed as one.**
   `test_lesson6_invalid_json_arguments_not_executed_validation_error_returned`,
   `test_lesson6_unknown_tool_name_not_executed_validation_error_returned`,
   `test_lesson6_schema_violation_extra_field_not_executed`,
   `test_lesson6_prose_that_looks_like_a_tool_call_is_never_parsed_or_executed`.

Cross-cutting behavior (LLM error events, cancellation, emitted UI events,
and the `TurnResult` logging fields `route`/`calc_calls`/
`research_calls`/`cached_tokens`) is covered by the remaining tests in the
same file.

## Validation path

`run_turn` is a pure function of its fake collaborators (`llm`,
`research_engine`, `calc`) plus `emit` (a callback receiving UI-facing
events: token deltas, tool-call/tool-result activity) and an optional
`threading.Event` for cancellation. This keeps the whole turn-flow
contract testable without a live llama-server or a real ZIM archive: the
real collaborators (`tutor.app.llm_client.LlamaClient`,
`tutor.retrieval.research.ResearchEngine`, `tutor.tools.calc_tool`) are
wired in by the production app entry point, not by this module.

## Turn pipeline as of 2026-09-21

With today's settings all at their new defaults (`app.host_topic_gate`,
`app.model_writes_search`, `app.model_may_skip_search`,
`app.reuse_prior_passages`, `app.restate_question_last`,
`app.no_specifics_without_source`, `app.child_safe_body_topics`), one
free-text turn runs through these stages in order:

1. **Host topic gate** (`tutor.app.topic_gate.classify_message`, gated by
   `app.host_topic_gate`) — pure, no-I/O classification of the raw
   student message into `"decline" | "chat" | "normal"`, before any
   search or LLM call. A `"decline"` verdict short-circuits the rest of
   this pipeline entirely: the host appends a fixed reply to the log and
   returns, zero LLM calls, zero research calls. A `"chat"` verdict skips
   straight to step 7 with a synthetic no-search tool result. Only
   `"normal"` continues to step 2.
2. **Raw pre-search** — the deterministic word-match pre-retrieve against
   the student's raw text always runs here (for cache/ordering reasons),
   but its result is discarded rather than used whenever
   `app.model_writes_search` is on and the forced call below produces
   something usable.
3. **Forced research call** (`app.model_writes_search`) — the model is
   asked, in one tool call, for an optional `question` (the turn restated
   as a standalone question, referents resolved), `queries` (short
   title-like search strings), and — when `app.model_may_skip_search` is
   on — a `needs_search` boolean. The decision table in
   `_run_forced_rewrite_round`/`run_turn` resolves this into skip
   (`needs_search: false`), search-by-question (no queries but a usable
   question), search-by-queries (the normal case), or fall back to the
   turn's own raw pre-search result (no queries, no question).
4. **Merge / relabel** — the forced round's results are RRF-merged with
   any earlier evidence and relabelled `[S#]` without duplicates
   (`5db7277`).
5. **Reuse pointers** (`app.reuse_prior_passages`) — evidence already
   held from an earlier turn in the same lesson is referenced by a
   pointer line instead of re-pasted in full, cutting prompt tokens
   (~16% by turn 6 in measurement).
6. **Evidence tail by level + restate line** — the tool-result text the
   model sees differs by evidence level (`strong` / `weak` / `empty` /
   `skipped`); `app.no_specifics_without_source` adds a no-invented-
   specifics line to every level including `strong`.
   `app.restate_question_last` appends the model's own resolved
   `question` (or nothing, if none was supplied) as a `(meaning: ...)`
   clause at the very end of the turn, after the evidence, so the model
   answers the actual current question rather than drifting onto its own
   search topic.
7. **Answer** — the model's real reply, generated once evidence/tail (or
   the topic-gate's synthetic chat/decline result) is in place.
8. **Host attribution / specifics gate / computed check** — sentence-
   level attribution (`attribute_sentences`) marks each sentence found/
   not-found/unbacked/computed independently of the model's own labels;
   the specifics gate (`is_supported`) requires every number/name a
   sentence states to appear verbatim in this turn's evidence once the
   sentence states a specific at all; `computed_check` re-verifies stated
   arithmetic and never fails the turn outright even on bad input
   (Unicode minus signs, `bfb9954`).
9. **Done event with timings** — the SSE `done` event carries
   `route` (`"declined"` / `"action"` / `"preretrieve"` /
   `"preretrieve+followup"`), `evidence` levels before/after
   (including the `"skipped"` value), and the `timings` dict used for
   all latency measurement (the only reliable per-stage timing source —
   `TestClient`-buffered SSE per-event timestamps are not).

The settings switching each stage: `app.host_topic_gate` (1),
`app.model_writes_search` + `app.model_may_skip_search` (3),
`app.reuse_prior_passages` (5), `app.no_specifics_without_source` +
`app.child_safe_body_topics` + `app.restate_question_last` (6),
`app.model_writes_citations` (whether the answer round asks for `[S#]`
labels at all, no longer on by default).

The system prompt used to build the first message of every turn lives at
`tutor/app/system_prompt.txt` (packaged via `pyproject.toml`
`[tool.setuptools.package-data]`): it instructs the model to teach rather
than answer outright, cite `[S#]` for source-backed statements, call
`calc` before asserting arithmetic beyond single digits and show the
expression evaluated, say so when giving its own (non-sourced) example,
treat `[Q&A]`-marked evidence as a contributor's answer rather than an
authoritative reference, and stay within the ~800-token generation budget
from the 32K profile in `tutor/app/prompt.py`.

# Forced query rewrite on weak evidence

When the host's pre-retrieval `assess_evidence` (tutor/retrieval/assessment.py)
comes back `weak`/`empty` for a factual (`preretrieve`) turn, the host
forces a query rewrite instead of trusting the model to volunteer a
search: measured, this LLM never does.

## Forcing mechanism + fallback

Primary: `LlamaClient.stream_chat(..., tool_choice={"type": "function",
"function": {"name": "research"}})`, added additively (OpenAI-compatible
field, passed through verbatim, never validated by the client). Fallback,
used only when that call yields a StreamEvent `kind == "error"` (server
rejected the field): a `response_format` json_schema request for
`{"queries": [...]}`, with the client's own code synthesizing an
equivalent tool call locally. Both paths are exercised with a fake
`LlamaClient` (no live server) in `tests/test_agent_loop_rewrite.py` and
`tests/test_llm_client.py`. `docs/dev_runtime.md` records the current
build's chat-template/tool-call capability but not a directly observed
`tool_choice` accept/reject result, so the fallback is a tested
compile-time guarantee, not an assumption the primary path always works.

## Host note placement

The note is appended once, in-line, as part of the single student user
message written to the `PromptLog` for this turn (`_REWRITE_HOST_NOTE` in
`tutor/app/agent_loop.py`) -- never edited in afterwards. This keeps every
log write append-only (required for a Mamba-hybrid model with no
prompt-cache forking) and keeps the chat template's role sequence
unchanged: it is still exactly one `user` message, just with a bracketed
host note as part of its own text, not a second message or a `system`
message injected mid-conversation.

## Merge / re-assess rule

The host executes each of the model's 1-3 rewritten queries against the
research engine, merges the resulting passage lists by keeping each
passage's best (lowest) rank across all lists, sorts best-first, caps to
`_MERGE_CAP` (8) passages, and re-runs `assess_evidence` against the
original question plus the rewritten queries' own terms.

## Not-found instruction

When the merged, re-assessed evidence is still `weak`/`empty`, the single
tool-result message appended to the log instructs the model: "tell the
student plainly that you could not find this in the library, suggest a
better way to ask or a related topic in the sources given so far, and --
only if it adds anything from memory -- keep it brief, clearly labelled as
from memory and unchecked, with no specific numbers/dates/names."

## Setting

`[app] rewrite_on_weak_evidence` (`AppConfig.rewrite_on_weak_evidence`),
default `True`. `False` reproduces today's byte-for-byte prompt (no host
note, no forced call, plain weak/empty evidence appended as before) --
see `test_setting_off_is_byte_identical_to_no_rewrite_support`.

## Additive fields

- `TurnResult.evidence` / SSE `done`+`attributions` events: `evidence:
  {level_before, level_after, rewritten_queries, corrected_terms}`
  (`None` for non-factual routes).
- UI (`tutor/ui/app.js`, `appendSearchedForLine`): a muted "Searched for:
  ..." line above the answer, plus one of two not-found notes keyed off
  `evidence.level_after` (`weak` vs `empty`), textContent only.
- Eval (`eval/run_lesson_soak.py`, `eval/run_turn_eval.py`): per-turn
  `evidence` field, and summary `rewrite_rate` / `rescued_rate`.

## Limits

At most one forced rewrite round per turn; never on `action` routes; never
when pre-search is already `strong`; bounded by the existing retrieval
deadline. `research` schema keeps `query` working unchanged and adds
`queries: string[]` (1-3, max 80 chars each) additively.

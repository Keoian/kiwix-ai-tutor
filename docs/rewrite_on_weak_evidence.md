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

The host executes the model's 1-3 rewritten queries as ONE batched call,
`ResearchEngine.research_many(queries, ...)` (`tutor/retrieval/
research.py`), which shares a single deadline across the whole batch and
this engine's own persistent per-archive worker pool / IDF/response
caches, running each query concurrently (per-query wall time is mostly
blocking IPC to the out-of-process ZIM worker, which releases the GIL --
see `docs/retrieval_baseline.md`'s profiling note -- so N queries cost
close to the slowest single query, not the sum of all of them). A
per-query failure or an already-elapsed shared deadline never raises or
drops a slot; it comes back as `{"status": "error", ...}` for that query
alone.

The resulting passage lists are merged by `_merge_dedupe_passages`
(`tutor/app/agent_loop.py`) via **reciprocal-rank fusion (RRF, k=60)**,
computed first at ARTICLE level (so an article found by >= 2 of the
rewritten queries reliably outranks one only a single query found, even
if that single-query hit had a better raw rank) and then at passage level
within each article's own rank slot. A **title-match boost** applies when
an article's own title terms are fully contained in one of the rewritten
queries (e.g. "Square foot gardening" ⊂ "square foot gardening basics");
a **penalty** applies to disambiguation pages and titles carrying only a
single, generic content term (e.g. "Garden", "Square (disambiguation)").
Still capped to `_MERGE_CAP` (8) passages; ties are broken deterministically
by (best original rank, title, passage id).

`assess_evidence` is then re-run with two new, backward-compatible
optional parameters: `rewritten_queries` (the union of each rewritten
query's own key content terms, replacing the original question's terms
for coverage purposes) and `healthy_terms` (any of the ORIGINAL question's
own key terms that already had a healthy match, kept alongside the
rewritten terms). This matters because the original question's
misspelt/fused terms (e.g. "squarefoot") can never be covered by good
passages -- checking them permanently pinned coverage low even when the
model's rewrite and the merged evidence were both good. See "Before/after"
below.

### Before/after (live smoke, `data/rewrite_smoke.py`)

- **Before this fix**: query "how to squarefoot garden the right way?" --
  Granite's three rewritten queries were all reasonable ("how to square
  foot garden the right way", "square foot gardening basics", "square
  foot gardening guide step by step"); the merge kept best-rank-per-passage
  and put the target article **last**: `['6 Foot 7 Foot', 'Garden',
  'Square foot', 'Square (disambiguation)', 'Square foot gardening']`.
  Re-assessment against the original ("squarefoot"-containing) terms
  stayed `weak`, so the tutor told the student it found nothing. Added
  wall time: 8.3 s (forced call 3.45 s + ~4.9 s for three sequential
  searches).
- **After this fix**: same question, live run against the real
  llama-server -- `level_after` is now `strong`, "Square foot gardening"
  is present with a real evidence passage (`Square foot gardening` in the
  merged/observed titles, `6 Foot 7 Foot`/`Square (disambiguation)` no
  longer crowd it out of the assessed evidence), and the answer's
  attributions are `backed=4 unbacked=0`. Prompt-cache survival is intact:
  turn 2's `cached_tokens` (1716) is within one token of turn 1's
  `total_tokens` (1717). (This particular live run, the model rewrote to
  a single query rather than three -- Granite's own rewrite choice is
  non-deterministic across runs -- so it exercises the assessment-terms
  fix directly and the RRF merge fix on a smaller (1-query) case; the RRF
  merge's multi-query behaviour is covered by
  `tests/test_agent_loop_rewrite.py::test_rrf_merge_puts_multi_query_target_article_first`,
  which reproduces the exact three-query/never-first-place shape from the
  original bug report and asserts "Square foot gardening" now sorts
  first.)

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

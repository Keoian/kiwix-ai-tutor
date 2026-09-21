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

### Follow-up: latency and merged-order fixes (live smoke, re-measured)

Two problems remained after the fix above landed, found by re-running
`data/rewrite_smoke.py` (now instrumented with per-LLM-request
`timings` -- llama.cpp's own `prompt_ms`/`predicted_ms`/`*_per_second`
block, captured additively onto `StreamEvent.usage` in
`tutor/app/llm_client.py` -- and per-`research()`/`research_many()`
elapsed timers) several times live against llama-server, since
Granite's own rewrite wording is non-deterministic run to run.

**Where the ~27 s outlier went / current timing.** `research_many`
already ran its queries concurrently in threads sharing one deadline
(confirmed empirically: a 3-query batch's own elapsed time, ~3.3 s, was
close to its single slowest query, not the ~9.7 s sum of the three) --
the earlier ~4.9 s "three sequential searches" estimate in the
Before/after note above was itself measured before this batching
landed. Repeated live runs after it land in the 9-17 s range for turn
1 (well under the original 27.3 s outlier, which looks like a cold-run
artifact -- no run since has approached it). The rewrite round's own
added cost is the forced tool-call request (~2-3 s) plus the batched
`research_many` call (~3-4 s, bounded by its slowest query) -- close to
the ~5 s target, and a large improvement on the ~8.3 s this doc
previously measured for the sequential-search path.

**Why "...basics" (and similar model-written rewrites) came back
completely empty.** Not a retrieval miss: `research()`'s own AND-of-terms
fallback (`_search_with_fallback`) already degrades gracefully and found
real candidates -- `square foot gardening basics` alone returns 5
full-text hits and 10 title hits once the exact-phrase AND query fails.
The candidates were found and packed, then silently discarded by
`research()`'s own coverage gate: `assess`-like coverage is computed
against the QUERY STRING'S OWN terms (`coverage_terms`), and a query the
model wrote by appending one more of its own words ("basics") to an
otherwise-good query can end up with a real, on-topic passage set that
doesn't itself contain that one extra word -- coverage looks "weak", and
the (perfectly good) passages get dropped to `[]` before ever reaching
`research_many`'s caller. This is exactly the class of problem the
merge/re-assessment already exists to solve one level up (in
`agent_loop`, via `rewritten_queries`/`healthy_terms`) -- but the
passages never got that far because `research()` zeroed them out first.
Fix: `ResearchEngine.research()` takes a new keyword-only
`relax_coverage_gate: bool = False` (default off, so the single/legacy
call path is unchanged byte-for-byte); `research_many()` passes
`relax_coverage_gate=True` for every one of its (already
model-authored, already-spelled) queries, which keeps the packed
passages even when that one query's own coverage looks weak, trusting
the caller's own merged re-assessment to judge real coverage. Verified
directly against the live archive: `research("square foot gardening
basics")` alone now returns real passages (`6 Foot 7 Foot`, `Square
foot`, `Garden`, `Companion planting`, ...) instead of `[]`.

**Final merged order.** Even once a query like "square foot gardening"
alone returned "Square foot gardening" as its OWN top-ranked passage
(rank 0 of 6, confirmed by direct engine call), `_merge_dedupe_passages`
still put it last behind `6 Foot 7 Foot` and `Square (disambiguation)`
in one single-query live run, and the RRF vote-count-first sort could let
`6 Foot 7 Foot` (found by 2 of 3 rewritten queries, since it happens to
share the word "foot") outrank the target (found by only 1) in a
three-query run. `_is_generic_or_disambiguation` only ever penalized
disambiguation pages and single-generic-term titles; a multi-word but
still purely-incidental title (a song sharing exactly one word with the
query, and none of its OTHER terms) was never penalized. Fix: it now
also flags a title as generic/incidental when it is NOT itself a subset
of any rewritten query's own terms (so a genuine `_TITLE_BOOST` subset
match is never double-penalized) AND its overlap with the union of all
rewritten queries' terms is at most one word. `_merge_dedupe_passages`'s
article sort now buckets by this flag FIRST, ahead of raw vote count, so
an incidental/generic hit can never outrank a real one purely on being
found by more queries. New regression test:
`tests/test_agent_loop_rewrite.py::test_rrf_merge_incidental_multi_hit_never_beats_single_query_target`,
reproducing the exact "6 Foot 7 Foot found by 2 queries vs. Square foot
gardening found by 1" shape from the live smoke run. Verified live:
a single-query "square foot gardening" rewrite now merges to
`['Square foot gardening', '6 Foot 7 Foot', 'Square (disambiguation)',
...]` (target first). Residual, out of scope for this fix: when the
model's OWN rewrite wording never retrieves the target article at all as
a candidate in the first place (e.g. it rewrites to the verb "garden"
rather than "gardening", which the archive's search does not stem
together), no merge-order fix can rank in a passage that was never a
candidate -- that is a retrieval-recall question, not a scoring one, and
the tuning-split recall/MRR numbers below confirm `research()`'s own
recall is unchanged by everything in this section.

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

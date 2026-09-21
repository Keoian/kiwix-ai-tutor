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

At most one forced rewrite round per turn on the weak-evidence path (see
"Follow-up rewrite" below for the separate, additional follow-up round);
never on `action` routes; never when pre-search is already `strong`
(except on a follow-up turn -- see below); bounded by the existing
retrieval deadline. `research` schema keeps `query` working unchanged and
adds `queries: string[]` (1-3, max 80 chars each) additively.

## Follow-up rewrite

Motivating transcript: in a lesson about DNA, Q1 "What's the largest
molecule?", Q2 "What about DNA?", Q3 "Is it a molecule?" -- the tutor
answered a generic "what is a molecule" essay for Q3, because the
deterministic pre-search ran on the raw text "Is it a molecule?" and
retrieved the "Molecule" article with `strong` evidence, so the
weak-evidence mechanism above never even looked at it: the evidence was
strong, just about the wrong thing.

**Design choice: no detector gates this.** An earlier version of this
fix (`tutor.app.followup.is_elliptical_followup`, still present as a
pure, tested helper but no longer wired into `agent_loop`) tried to
detect "this question only makes sense in context" via pronouns
("it"/"that"/...), bare openers ("what about", "why"), and a low
content-term count. It was dropped as the gate: a real student's own
grammar or spelling can't be relied on to signal an unresolved reference
reliably enough to gate a correctness fix on. Instead, **every turn after
the lesson's first** forces a follow-up rewrite round, unconditionally
(`[app] rewrite_on_followup`, default `True`), regardless of how strong
the raw pre-search looks.

**Mechanism.** The raw pre-search (on the literal question text) still
always runs first, exactly as before -- its passages are never discarded,
only demoted. Before the model ever sees that raw evidence, the host
forces one `research` tool call (`_run_forced_rewrite_round`, the same
`_forced_research_tool_call` primitive as the weak-evidence path above)
using a follow-up-specific host note (`_FOLLOWUP_HOST_NOTE`): write 1-3
short standalone queries that resolve any pronoun/reference using the
lesson so far, and fix spelling. The rewrite's own query results are
merged via `_lead_with_backfill` -- the rewrite's own passages LEAD in
rank order, and the raw pre-search's passages only fill remaining slots
behind them (never RRF-blended as equals, unlike the weak-evidence path's
own-query merge): the raw pre-search is demoted to backfill, not
discarded, since it can still be a correct answer to a question that
happens not to be elliptical (e.g. turn 2 asking a brand new, fully
standalone question). If the merged result is still `weak`/`empty`, the
existing weak-evidence mechanism gets exactly one more forced round on
top (capped: never a third round) -- otherwise the merged evidence is
used directly, with a short instruction appended after it
(`_FOLLOWUP_DIRECTNESS_NOTE`): answer the student's actual question
directly first (yes/no, if it's a yes/no question), then explain.

Turn 1 is unaffected (no prior turns to reference): the raw pre-search
and the existing weak-evidence mechanism run exactly as documented above,
byte-for-byte.

**Cost.** One extra forced tool-call round (append-only, prompt-cache
preserving, same mechanism/latency profile as the weak-evidence round)
plus one extra batched `research_many` call, on every turn after the
first, not only "weak" ones -- see the live smoke measurement below for
the added wall-clock cost this trades for correctness on a follow-up
turn.

**Setting.** `[app] rewrite_on_followup` (`AppConfig.rewrite_on_followup`),
default `True`. `False` disables only this mechanism; the weak-evidence
rewrite above is unaffected and still applies to every turn including the
first.

**Tests.** `tests/test_agent_loop_followup.py`: fires on turn 2 even when
the raw pre-search is independently `strong` (asserting the rewrite's own
passage leads the raw pre-search's passage in the rendered evidence
text); never fires on turn 1; disabled cleanly by the setting. Detector
unit tests for the retained-but-unused `is_elliptical_followup` helper
are in `tests/test_followup.py`.

### Live smoke (`data/rewrite_followup_smoke.py`)

Run in-process (build_deps + TestClient, `config/archives.simplewiki_only.toml`)
against the real dev llama-server already running on :8080, comparing
`rewrite_on_followup` ON vs OFF for the exact motivating 3-turn lesson
(Q1 "What's the largest molecule?", Q2 "What about DNA?", Q3 "Is it a
molecule?"):

- **ON**: turn 3 forced a rewrite round every time, rewriting to `"Is DNA
  a molecule?"` (turns 2 and 3 both rewrote to this same query); turn 3
  answer's first sentence: "Yes, DNA (Deoxyribonucleic Acid) is a
  molecule." -- direct yes/no first, as instructed.
- **OFF**: turn 3's raw pre-search on "Is it a molecule?" happened, on
  this particular archive/model run, to still land on DNA-relevant
  evidence (no rewrite fired at all, `rewritten_queries: []`), giving a
  similarly correct first sentence ("Yes, DNA is a molecule."). This
  does **not** contradict the original bug report -- the original
  failure (a generic "what is a molecule" essay) depended on the raw
  pre-search retrieving the wrong article, which is exactly the failure
  mode this mechanism removes reliance on, not one guaranteed to
  reproduce identically on every run/archive/model sampling.
- **Added wall-clock cost, turn 3, ON vs OFF (single run, this
  archive/model)**: ON 16.30 s vs OFF 5.56 s -- **+10.74 s** for the
  extra forced tool-call round plus its `research_many` batch. This is a
  single measurement, not an average; expect run-to-run variance similar
  to the weak-evidence path's own measured 9-17 s range for a rewrite
  round (see "Follow-up: latency and merged-order fixes" above).

**Not verified live**: a case where the raw pre-search's `strong`
evidence is demonstrably WRONG-TOPIC on this exact archive/model
combination (to directly reproduce the original bug's wrong-answer
outcome end-to-end); the cascade into a second (weak-evidence) round
after a weak follow-up-rewrite merge; behaviour with `rewrite_on_followup`
default (ON) against a longer, multi-topic lesson.

### Follow-up rewrite latency profile (`data/followup_latency_profile.py`)

The single-run smoke above measured **+10.74 s** on turn 3 (ON vs OFF),
well above the ~1-2 s expected for one ~20-token forced tool call on a
cached prompt plus a ~1 s search. To find where the time actually goes,
`data/followup_latency_profile.py` extends the smoke script: it
monkeypatches `LlamaClient.stream_chat` (class-level, since `AppDeps`
doesn't expose the `llm` instance) and `research_engine.research` to
record, per LLM call, `prompt_tokens`/`cached_tokens`/`prompt_ms` and
`completion_tokens`/`predicted_ms` from llama-server's `usage`/`timings`,
plus wall time per retrieval call. Run against the same dev llama-server
on :8080, 3 reps ON and 3 reps OFF, same 3-turn lesson, checkpointed to
`data/followup_latency_profile_20260921_075137.json`.

**Turn 3 wall time, 3 reps each (seconds):**

| | rep0 | rep1 | rep2 | median |
|---|---|---|---|---|
| ON | 8.02 | 7.08 | 10.66 | **8.02** |
| OFF | 3.52 | 6.56 | 51.81\* | **6.56** |

\*OFF rep2 hit `max_tokens=2000` (a 2000-token runaway completion,
`finish_reason` capped) -- a sampling-variance outlier unrelated to the
forced-rewrite feature; included for transparency but excluded from the
breakdown below.

**Median added wall time, ON - OFF: ~1.45 s** -- in the expected 1-2 s
range, not the +10.74 s the single earlier smoke run showed. Per-rep
added time was highly variable (+4.50 s, +0.52 s, -41.16 s) because
answer length (`completion_tokens`, unconstrained, temperature > 0)
swings by 2-3x run to run regardless of the setting; the original single
ON/OFF sample simply landed on an unlucky pair.

**Breakdown of the forced round's own fixed cost (turn 3, reps 0-1,
median), measured not inferred:**

| Component | Measured | Suspicion status |
|---|---|---|
| Forced tool-call round (`_run_forced_rewrite_round`) wall | 1.31 s | confirmed real, small |
| \| prompt cache hit ratio on that round | 2919-3155 / 3045-3901 cached (~95-96% of that call's own 126-token prompt) | (a) prompt cache breakage: **refuted** -- forced round reuses the cached prefix, does not reprefill |
| \| its `completion_tokens` | 29-33 (capped by `max_tokens=96`) | (b) unbounded generation: **refuted** -- already capped, generates far under the cap |
| Extra `research()` call (2nd query) wall | 0.0-0.8 s | (d) 2nd search is slow: **refuted** -- comparable to or cheaper than the single OFF search (0.45-0.7 s) |
| Extra prompt prefill on the final answer call from merged evidence (`prompt_n` 746 ON vs 425 OFF tokens; `prompt_ms` 2168 ON vs 1481 OFF median) | **+0.69 s** | real, previously unmeasured cost: the forced round's extra evidence enlarges the final answer's own prompt, and a larger fraction of it is new (uncached) text |
| Answer length (`completion_tokens`) ON vs OFF, same rep | 186/84, 146/215, 286/2000 | (c) ON answers are simply longer: **refuted as a systematic effect** -- not consistently longer than OFF; sampling variance dominates and swamps any (a)/(b)/(d) signal in raw single-sample wall time |

Sum of the two confirmed fixed-cost components: 1.31 s (tool round) +
0.69 s (extra prefill) ≈ **2.0 s**, consistent with the 1-2 s expectation
once answer-length noise is averaged out via medians.

**Hypothesis (e), a second weak-evidence round firing**: **confirmed** as
a real, intermittent, larger cost. Turn 2 of ON rep2 fired 4 LLM calls
instead of the usual 2 (forced tool-call round -> its own weak-evidence
fallback text round -> a normal auto tool-call round -> the final
answer), adding roughly 1.2 + 1.6 + 1.3 s ≈ **+4.1 s** beyond the normal
single-round turn in that one rep. This did not occur in the profiled
turn-3 reps, but is the largest single per-turn cost this profile
surfaced when it does fire, and is a more plausible partial explanation
for occasional large outliers (like the original +10.74 s sample) than
cache breakage or an unbounded forced call.

**Recommended fix**: no cache or `max_tokens` bug to fix on the forced
round itself (a, b already correctly bounded/cached). Two real,
worth-fixing costs: (1) the extra ~0.7 s prefill from merged evidence on
the final answer -- consider capping/truncating the forced round's
contribution to the merged evidence context size closer to what a
single-query search would have contributed, rather than concatenating
both; (2) the intermittent second (weak-evidence) round after a weak
forced-rewrite merge (~+4 s when it fires) -- consider skipping that
second round specifically when it was already reached via the forced
follow-up round (i.e., treat the forced round's own weak result as
terminal rather than triggering a further escalation), since the forced
round was itself already the escalation.

# Code review pass 2 — 2026-09-20

Scope: everything added since `docs/review_2026-09-20.md` (see its Resolution section):
`tutor/app/{agent_loop,session,prompt,compose,routes,main,citations,profiles,lesson_state,
resources,turn_log,source_view,research_tool}.py`, `tutor/ui/*`, `tutor/retrieval/research.py`
(coverage flags, topic_hint, tier-2 gating, dense fusion, ranking flags),
`tutor/retrieval/hybrid/{dense,ranking}.py`, `tutor/retrieval/index/*`, `eval/*.py`,
`eval/tools/*.py`, `scripts/*`. Judged against
`docs/plan/offline_tutor_implementation_plan.md` §5/§10 and
`docs/plan/offline_tutor_spec_v0.3.md` §6/§8 (as superseded by plan §0.2)/§9/§11/§12.

## Findings, most severe first

### 1. A tool-call exception mid-turn leaves the session's `PromptLog` with a dangling `tool_calls` assistant message and no matching tool result — the session is silently and permanently wedged
- **Where:** `tutor/app/agent_loop.py:303-304` appends the assistant's `tool_calls` message to
  `session.log` (`log.append_assistant_tool_calls(wire_tool_calls)`) *before* the dispatch loop
  at `tutor/app/agent_loop.py:318-369` runs each call. If `research_engine.research(...)`
  (`agent_loop.py:334-338`) or `calc.evaluate(...)` (`agent_loop.py:361`) raises for any call
  after the first one in a multi-tool-call turn (or the first one), the exception propagates out
  of `run_turn` uncaught — there is no `try/except` anywhere in the dispatch loop.
- **What catches it, and what doesn't:** `tutor/app/compose.py:240-253`'s `turn_runner` wraps
  `run_turn(...)` in `try/except Exception` and emits a student-safe `error` frame, so the
  *client* sees a clean error — but `session` (holding the mutated `PromptLog`) is a long-lived
  object in `_SessionStore` (`compose.py`), and the exception is caught *after* the log mutation
  already happened. The half-written turn is never rolled back.
- **Concrete failing scenario:** student asks a question, model calls `research`, the archive
  worker times out or raises (a real possibility — `ZimWorker`/`_call_worker` in
  `tutor/retrieval/research.py` do hit deadlines under I/O pressure per the first review's
  finding #3, and any exception not swallowed by that layer propagates here). `log` now ends
  with `..., {"role":"assistant","tool_calls":[...]}` and no `{"role":"tool","tool_call_id":...}`
  entries. The *next* turn in the same session calls `log.render()` /
  `_to_wire_messages(log.render())` (`agent_loop.py:216`) and sends that message list to
  `llama-server`'s `/v1/chat/completions`. OpenAI-compatible endpoints require every
  `tool_calls` assistant message to be immediately followed by tool-role messages for each id;
  llama-server is very likely to 400 on this, and the session is now unusable for the rest of
  its life (including after a lesson resume, since `LessonStore.save_session` persists
  `log.render()` verbatim — the corruption is durable, not just in-memory).
- **Fix:** wrap the per-call dispatch body (`agent_loop.py:318-369`) in `try/except`, and on any
  exception append a synthetic error tool result for *every* tool-call id in the current
  assistant turn that hasn't yet gotten one (not just skip and re-raise), before propagating (or
  swallowing) the error — so `log` never contains an assistant `tool_calls` message without a
  matching tool result for each id. Add a test that raises inside a fake `research_engine`
  mid-turn and asserts `log.render()` is still a valid, replayable OpenAI message list afterward
  (every `tool_calls` message immediately followed by its tool results).

### 2. `topic_hint` tokens are folded into the same term set used for the abstention/coverage gate, so a subject hint alone — with zero overlap between the passage and the actual student question — can make an off-topic candidate "covered"
- **Where:** `tutor/retrieval/research.py:87-96` (`_query_terms`) unions `tokenize(topic_hint)`
  into the same `query_terms` set that `_query_terms`'s caller then passes straight into
  `_best_coverage`/`compute_coverage` (`research.py:614`, `698-704`, `107-153`).
  `compute_coverage` (`research.py:133-153`) treats `query_terms` as an undifferentiated bag:
  `term_coverage = len(query_terms & passage_terms) / len(query_terms)` and
  `title_match = bool(query_terms & title_terms)` — there is no way to tell, from inside
  `compute_coverage`, whether a term matched because of the student's actual words or only
  because of the host-supplied subject hint.
- **Why wrong:** spec §6's "tier 2 only if coverage flags are weak after tier 1" and §7.3's
  "coverage signals are explainable" assume `weak` reflects whether the *query* is answered by
  the *candidate* — not whether the current subject happens to be mentioned in it. As written,
  a candidate whose title/text matches only the topic hint token(s) (not any content word of the
  actual question) is scored non-weak, so the response is returned with `status="ok"`/`"partial"`
  and real, snapped-shot passages — i.e. the model is handed "supporting evidence" for a
  question it doesn't actually support, instead of the abstention/tier-2-fallback path the
  spec intends when coverage is genuinely weak.
- **Concrete failing scenario:** `topic_hint="Photosynthesis"`, student asks (off-topic, e.g. a
  test typo or a genuinely unrelated tangent) "what's the capital of France?" — if any indexed
  article's title or body contains the word "photosynthesis" (near-certain in a science archive
  restricted by `_route`/`for_subject`) and shares zero words with "capital of France",
  `title_match` is still `True` purely off the hint token, `weak=False`, and passages about
  photosynthesis are packed and returned as if they answered the France question.
- **Fix:** compute `title_match`/`term_coverage` from the query's own content-word tokens only;
  keep the topic-hint tokens as a separate, weaker signal (e.g. only used to break ties among
  otherwise-qualifying candidates, or folded in with a discount), not unioned directly into the
  hard weak/non-weak boundary. Add a coverage unit test with a topic hint that matches a
  candidate but the query text does not, asserting `weak is True`.

### 3. `Budget.scaled()`'s fixed-overhead-first split starves `history` non-linearly at small ceilings — confirmed **not a production bug, but the M5 note's "should scale to ~4000" claim is simply wrong arithmetic** (verdict for investigation item 1, with numbers)
- **Where:** `tutor/app/prompt.py:66-93`. `fixed = system(800) + newest(2500) + generation(2000)
  = 5300` is subtracted from the ceiling first; only the *remainder* is split 80/20 between
  `history`/`margin`. `PromptLog.evict` (`prompt.py:242-285`) compares `tokens_used()` (the
  **whole** rendered log, including the system message) against `operating_ceiling =
  budget.system + budget.history` — i.e. it re-adds `system` on top of `history` as if `history`
  already excluded it, while `tokens_used()` counts the system message a second time.
- **Numbers, run directly:**
  ```
  Budget.scaled(6000)  -> system=800 history=560  newest=2500 generation=2000 margin=140
                          operating_ceiling = 800+560 = 1360
  Budget.scaled(8192)  -> history=2316,  operating_ceiling=3116
  Budget.scaled(16384) -> history=8877,  operating_ceiling=9677
  Budget.scaled(32768) -> history=22000, operating_ceiling=22800   (== the hand-tuned default)
  ```
  At the M5 test's 6,000-token ceiling, the real eviction threshold is **1,360 tokens**, not the
  "~4,000" the M5 note assumed — so 48/60 evictions with max `tokens_used` ≈ 1,318 (just under
  the real 1,360 threshold, since the reported max is measured *after* eviction runs) is exactly
  what the code does, not an artefact of a miscounted fake token counter or of eviction "running
  every turn" independent of the numbers. `docs/M5_notes.md`'s inline comment ("a small ceiling
  (e.g. 6K) yields a small history allowance, which is exactly what the eviction unit tests rely
  on to force eviction") in `prompt.py:76-78` shows this is *intentional* for tests. Production
  runs at `cfg.server.ctx_size = 32768` (`config/dev.toml:14`), where `Budget.scaled(32768)`
  reduces exactly to the hand-tuned default (`history=22000`) — no starvation there.
  **Verdict: artefact of an unrealistically small 6,000-token test ceiling combined with the
  scaling formula's fixed-overhead-first design (not a bug in `evict()`'s trigger logic, and not
  a problem at the real 32,768 ctx_size), but the M5 note's "~4,000" expectation is arithmetically
  wrong and should be corrected so it doesn't mislead a future reader into "fixing" a non-bug.**
- **Residual concern:** if `ctx_size` is ever configured below roughly 7,300 (`fixed` + a
  reasonable margin) for weaker hardware, `history` collapses much faster than the 80/20 ratio
  suggests (e.g. 8192 → history=2316, a ~64% cut from the naive expectation of ~5,754), which is
  worth a comment in `prompt.py` next to `scaled()` so nobody is surprised by a real (not just a
  6K-test) small-ctx_size deployment.
- **Fix (optional, low urgency):** either note the non-linearity explicitly in the `scaled()`
  docstring with a worked small-ceiling example, or change `evict()`'s `operating_ceiling` to
  `budget.ceiling - budget.newest - budget.generation - budget.margin` (equivalent to
  `budget.system + budget.history` by construction, so purely a documentation/clarity fix, not a
  behavior change) so the "system counted twice" appearance goes away.

### 4. `ResearchEngine._response_cache` is an unbounded, unkeyed-by-time `dict` — grows for the life of the process
- **Where:** `tutor/retrieval/research.py:317` (`self._response_cache: dict[...] = {}`),
  written at `research.py:744`, read at `research.py:588`. No `maxsize`, no LRU eviction, no TTL,
  and the key includes `query` (raw free text) — every distinct question asked across every
  session for the process's lifetime adds one more permanent entry holding a full
  `ResearchResponse` (passages, coverage dict, timings).
- **Impact:** on a long-running host (the design target is a persistent local server, not a
  short CLI invocation), this is unbounded growth proportional to the number of distinct
  questions ever asked — the plan's resource-discipline goals (§5/§10, `ResourceMonitor`) are
  about exactly this class of leak, and nothing here is measured or bounded.
  Also: raw student free-text (`query`) now lives indefinitely in the cache key, in memory,
  when spec §11/§12 privacy intent is that turn text isn't durably retained outside the
  lesson's own `turns` table (see item 7 below — this cache is in-process/RAM only, so it isn't
  a `turns.jsonl`/logging violation, but it is an unbounded, silently-retained copy of every
  question ever asked, and it would appear in a heap dump / restart-persisted memory profile if
  this process runs for weeks as intended).
- **Fix:** bound it (e.g. `functools.lru_cache`-style with a `maxsize`, or an explicit
  `OrderedDict` capped and evicted LRU-style), and consider excluding raw `query` text from the
  key material retained if this is meant to survive review under §11/§12 privacy intent.

## Checked and fine
- **UI XSS:** `tutor/ui/app.js` — grep for `innerHTML`, `insertAdjacentHTML`, `eval(`,
  `new Function`, `document.write` found none; every DOM text assignment found uses
  `.textContent` (lines 35, 57, 127, 297, 351/356/363/369, 411, and a code comment at 146
  explicitly documents the "textContent only" policy). No dynamic script/style injection found.
- **Privacy / logging:** `tutor/app/turn_log.py`'s `TurnLogger.log_turn` has no
  `user_text`/`text`/`message` parameter (confirmed by signature and by `docs/M5_notes.md`'s own
  audit) so raw student free text is structurally unable to reach `logs/turns.jsonl`. Grepped
  every `tutor/app/*.py` for `logger.`/`logging.` calls: **zero** — the app layer does not use
  Python `logging` at all. `tutor/retrieval/research.py`'s one `logger.info(...)` call
  (`research.py:463-472`, `event=archive_searched`) logs `archive_id`, `storage`, `tier`, hit
  counts, and `timed_out` only — no query text, confirming storage class is present as asked.
- **Dense fusion vs. abstention:** `_best_coverage`/`compute_coverage` run the identical
  lexical term-overlap check against a dense-only candidate's real archive `title`/`text` as
  against a lexical candidate — dense routing does not bypass or weaken the coverage gate itself
  (only `topic_hint`, item 2 above, does). Embedding the query (`research.py:617-636`) is
  correctly bounded by `min(_EMBED_QUERY_TIMEOUT_S, soft_remaining)` against the soft deadline
  and degrades to lexical-only on any failure/timeout without failing the whole request.
- **Tier-2 gating:** `primary_archives`/`fallback_archives` split by `entry.tier != 2` /
  `== 2` (`research.py:608-609`) matches spec §6's "tier 1 → tier 2 only if weak"; tier-3 is
  filtered earlier in `_route` alongside tier 1 as intended.
- **Calc/research caps, cap-reached tool results, unknown-tool/malformed-JSON handling:**
  unchanged from the first review's "checked and fine" list; re-spot-checked, still correct.
- **Cancellation:** `run_turn`'s cancel check (`agent_loop.py:196-204`) fires before each model
  call and `finish_reason == "cancelled"` mid-stream (`agent_loop.py:264-272`) both return a
  clean `status="cancelled"` `TurnResult` with no log mutation left dangling in either of those
  two paths specifically (only the exception path in item 1 leaves a dangling entry).
- **`routes.py` concurrency:** the `turns_lock`-guarded `active_turns` dict, per-turn dedicated
  thread, sentinel-terminated `queue.Queue`, and delete-only-if-still-mine cleanup
  (`routes.py:71-105`) are unchanged from the first review and still correct on inspection.
- **Xfail'd live tests:** `tests/test_turn_live.py`'s three `pytest.xfail(...)` calls
  (citation-rate, calc-invocation, calc-answer-surfacing) all carry a specific measured
  rate/date/reference (e.g. "measured 3/5 (60%) on 2026-09-20 ... see
  docs/citation_experiment.md") rather than a bare unexplained xfail — these satisfy "xfail with
  a measured reason," though the underlying 60% citation rate is itself a product-quality
  concern (below the stated 80% bar), not a review defect. These and `test_llm_live.py`'s
  `pytest.skip` on an unreachable `/health` are live-server-dependent by design and will not run
  (skip, not flake) on a CI runner without `llama-server`.
- **Windows/Linux script pairs:** `scripts/{serve,serve_dev,serve_embed,run_app,kiosk}.{sh,ps1}`
  both resolve the binary/argv via `python -m tutor.settings --argv <config>` (no
  flag-building duplicated per-OS) and both fail loudly (`set -euo pipefail` / `$ErrorActionPreference
  = "Stop"` + explicit `Test-Path`/`[ -e ]` checks) rather than silently continuing on a missing
  binary. No `select()`/signal-specific logic found in these scripts to diverge between OSes.

## Not fully investigated (time-boxed out of this pass — flag for a follow-up)
- Item 3 (lesson resume byte-identical round-trip including tool-call entries) and item 4
  (SQLite concurrent-request behavior, `ResourceMonitor` per-call cost) were reviewed by code
  inspection only (session's own `tests/test_lesson_state.py::TestResumeRebuildsByteIdenticalLog`
  already exercises the tool-call-entry round trip per `docs/M5_notes.md`, and appeared correct
  on reading `LessonStore.resume`/`_rebuild_prompt_log`), but no new script was run against them
  in this pass beyond what's cited above — recommend a dedicated follow-up if item 1's log-
  corruption fix (above) changes `PromptLog`'s append surface.
- `eval/*.py`, `eval/tools/*.py`, and `tutor/retrieval/index/*` were skimmed for the ranking-flag
  and coverage-flag wiring cited in items 2/4 above but not exhaustively reviewed line-by-line
  for every ranking flag combination.

## Resolution (2026-09-20, pass-2 fixes)

1. **Fixed.** `tutor/app/agent_loop.py`'s tool-dispatch loop now wraps each call in
   `try/except Exception`, appending a synthetic `{"ok": false, "error": ...}` tool result for
   that call's id on any exception instead of letting it propagate (the exception is swallowed,
   not re-raised, so the turn can still answer with what it has -- the append-only log is what
   must never be left invalid, not the exception itself). A cancel check was added *inside* the
   per-call dispatch loop (not just at the top of the outer while-loop): if `cancel` fires between
   two tool calls in the same turn, every remaining not-yet-dispatched call in that turn gets a
   synthetic `{"ok": false, "error": "turn cancelled"}` result before returning
   `status="cancelled"`, so a cancel mid-tool-loop can no longer leave a dangling entry either.
   Added `PromptLog.validate()` (`tutor/app/prompt.py`): raises `ValueError` unless every
   assistant `tool_calls` entry is matched by exactly its own tool results (tolerating a research
   follow-up's id-less evidence-packet tool message, which was already a valid existing shape --
   see the "assert best is not None"-adjacent code, unrelated). `PromptLog.repair()` detects a
   dangling trailing `tool_calls` entry and appends synthetic error results for the missing ids;
   `LessonStore.resume` now calls `log.repair()` and persists the repair back
   (`self.save_session`) if one was needed, so an already-broken persisted lesson (e.g. from
   before this fix existed) becomes usable again instead of staying wedged forever. Tests:
   `tests/test_review_pass2_fixes.py::test_tool_raises_leaves_log_valid_and_second_turn_succeeds`,
   `test_llm_error_mid_stream_after_a_tool_call_leaves_log_valid`,
   `test_cancel_mid_tool_loop_leaves_log_valid_and_second_turn_succeeds`,
   `test_prompt_log_validate_rejects_dangling_tool_calls`,
   `test_prompt_log_repair_fixes_dangling_tool_calls`,
   `test_lesson_store_resume_repairs_broken_persisted_log` -- each asserts `log.validate()` does
   not raise after the failure scenario, then runs a second normal turn and asserts it succeeds.
2. **Fixed, with a documented trade-off.** `tutor/retrieval/research.py`'s `_coverage_terms`
   now returns the query's own content terms only, except when the query is "elliptical" (at most
   one content term once a small set of continuation fillers -- "tell", "more", "explain",
   "elaborate", "continue" -- are also stripped), in which case the topic_hint's terms are folded
   in too. `_query_terms` (candidate generation/ranking) is unchanged. `_best_coverage` no longer
   receives a topic_hint-unioned term set implicitly; it just uses whatever `_coverage_terms`
   already decided. Tests added per the ask: off-topic query + matching topic_hint -> still
   `"empty"` (`test_off_topic_question_with_matching_topic_hint_is_still_empty`), elliptical query
   + topic_hint -> still finds the article (`test_elliptical_question_with_topic_hint_still_finds_article`),
   plus unit tests on `_coverage_terms`/`_best_coverage` directly. Tuning-split re-run (see
   `docs/retrieval_baseline.md`'s new "Pass-2 fix" subsection for the full before/after table):
   overall recall@5 0.567 -> 0.533, `elliptical` 1.000 -> 0.000 (regression, accepted and
   explained -- the tuning split's two `elliptical` items are richer multi-word phrasings than a
   bare pronoun reference and no term-count threshold can admit them without also re-opening the
   exact off-topic hole this fix closes; closing that gap for real needs semantic/dense retrieval
   over conversation context, not a lexical heuristic), every other category unchanged. Held-out
   split was not re-run (per instruction).
3. **Fixed.** `ResearchEngine._response_cache` is now an `OrderedDict` bounded at
   `_RESPONSE_CACHE_MAXSIZE = 256` entries, evicted LRU-style (`move_to_end` on hit and on
   insert, `popitem(last=False)` once over the cap). Test:
   `tests/test_review_pass2_fixes.py::test_response_cache_is_bounded`.
4. **Fixed (documentation only, as the finding itself concluded this is not a production bug).**
   `docs/M5_notes.md`'s "Simulated 30-minute lesson" section now carries a correction with the
   reviewer's own worked numbers: `Budget.scaled(6000)` yields `history=560`
   (`operating_ceiling=1360`), not "~4,000"; at the real 32,768 ctx_size `history` reduces to the
   hand-tuned default (`22000`). The note now states plainly that the small-ceiling test exercises
   eviction *mechanics*, not a realistic eviction *frequency*.

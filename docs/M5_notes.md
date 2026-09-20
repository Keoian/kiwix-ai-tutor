# M5 notes: profiles, lesson state, resource discipline, status/logging

## Data model

**`profiles.sqlite`** (`tutor/app/profiles.py`, `ProfileStore`), WAL mode,
`check_same_thread=False` + an internal lock:

```
profiles(id TEXT PK, display_name TEXT, grade_level INTEGER,
         subjects TEXT (JSON list), reading_level TEXT, preferences TEXT NULL)
```

**`lessons.sqlite`** (`tutor/app/lesson_state.py`, `LessonStore`), same
SQLite discipline:

```
lessons(id TEXT PK, profile_id TEXT, subject TEXT, ended INTEGER,
        prompt_log_json TEXT NULL)   -- the exact PromptLog.render() list
turns(id TEXT PK, lesson_id TEXT, subject TEXT, user_text TEXT NULL,
      action TEXT NULL, route TEXT, calc_calls INTEGER, research_calls INTEGER,
      citation_passage_ids TEXT (JSON list), tokens_used INTEGER,
      cached_tokens INTEGER, eviction_events TEXT (JSON list) NULL, seq INTEGER)
```

**`logs/turns.jsonl`** (`tutor/app/turn_log.py`, `TurnLogger`): one
compact, sorted-key JSON object per line, appended with `encoding="utf-8"`.

## What is (and is not) stored about a student

Stored: a display name, a numeric grade level, a list of subjects, a
reading-level tag, and an optional free-text `preferences` string
(`ProfileStore.create`/`update` reject any other kwarg with `TypeError`,
so nothing outside this field set can ever be persisted). This mirrors
spec §11 ("learner state kept small and explicit") and the citation-
snapshot model, which already treats extracted archive text -- not raw
model chat transcripts -- as the durable record.

Not stored: home address or any other PII, the student's raw per-turn
question text in the metrics log (`turns.jsonl`; `TurnLogger.log_turn`
has no `user_text`/`text`/`message` parameter at all, so it is
structurally impossible to log it there), and no field beyond
`display_name`/`grade_level`/`subjects`/`reading_level`/`preferences` on
`Profile` itself. The lesson-level `turns` table *does* keep
`user_text`/`action` (needed to answer "what did we cover last lesson"
type resume UI questions and matching the spec's lesson-as-archival-unit
model), but the separate, longer-lived JSON-lines operational log never
does.

## Resume semantics: byte-identical prompt log

`LessonStore.save_session(lesson_id, session)` serializes
`session.log.render()` (the flat message list `tutor.app.prompt.PromptLog`
produces) to JSON and stores it on the lesson row.
`LessonStore.resume(lesson_id, count_tokens=...)` reads that JSON back and
reconstructs a `PromptLog` (`_rebuild_prompt_log`) whose `render()`
reproduces the same entries: system message, protected (post-eviction
reprefill) passages, then turns split at each `user` entry.

`tests/test_lesson_state.py::TestResumeRebuildsByteIdenticalLog` asserts
`serialize_messages(session.log.render())` is byte-identical before/after
a save+resume round trip, survives a full process restart (new
`LessonStore` instance over the same `.sqlite3` file), and that the
append-only byte-prefix property (`tutor/app/prompt.py`'s central
invariant) still holds for turns appended *after* a resume -- i.e. resume
never rewrites or reorders anything already rendered, it only lets new
appends continue from the reconstructed state.

## Agent loop now reads/writes the session's append-only PromptLog

Fixed for M5: `tutor.app.agent_loop.run_turn` no longer builds a
throwaway local `messages` list every turn. When `session.log` is a real
`tutor.app.prompt.PromptLog` (as `tutor.app.session.Session` always
provides), `run_turn` appends the system prompt (once per session),
each user turn, evidence packets (deduped by passage id via
`PromptLog.append_evidence`), assistant tool-call turns
(`PromptLog.append_assistant_tool_calls`, OpenAI wire shape --
`{"id","type":"function","function":{"name","arguments"}}`), tool
results (`PromptLog.append_tool_result`), and final answers
(`PromptLog.append_assistant`, with `[S#]` labels extracted via
`tutor.app.citations.extract_labels` as `cited_labels` for eviction
protection) directly onto `session.log`. Before each model call,
`log.evict(budget)` runs and, if it fires, the resulting
`EvictionEvent` is both emitted (`{"kind": "eviction_reprefill", ...}`)
and carried on `TurnResult.events` for the turn logger. A new
`_to_wire_messages` translation strips internal-only bookkeeping
(`passages`, `cited_labels`) from `PromptLog.render()`'s output into the
plain OpenAI chat-message shapes the live `llama-server` endpoint
accepts -- the lack of this translation was the actual cause of a live
400 the first time this was wired up against the real server (evidence
messages have no `content` field internally; assistant messages carry an
extra `cited_labels` field). Sessions/fakes without a `.log` attribute
(most of `tests/test_agent_loop.py`'s ~30 tests) fall back to the
original local-list behavior unchanged, so that suite stays green with
no assertion changes. An oversized newest evidence packet is trimmed
(lowest-ranked passages dropped first, via `_trim_and_append_evidence`)
rather than raising `PromptOverflow` into the turn. Lesson resume
(`LessonStore.resume`) already reconstructed a `PromptLog` from
persisted `render()` output; because `run_turn` now reads/writes that
same object, resuming a lesson and calling `run_turn` again correctly
continues the same append-only log.

New test coverage: `tests/test_agent_loop_promptlog.py` (8 tests) checks
log entry ordering, the turn-to-turn byte-prefix property on the actual
messages sent to a scripted fake LLM, eviction firing before the model
call under a 6000-token ceiling with tokens-sent bounded by
`ceiling - generation - margin`, cited-evidence protection across
eviction, no duplicate evidence text for a repeated passage id,
action turns appending no evidence, oversized-packet trimming instead of
a crash, and that `TurnLogger` receives `eviction_reprefill=True` on an
evicting turn.

## Simulated 30-minute lesson (now through the real `run_turn`)

`tests/test_resource_discipline.py::TestSimulatedThirtyMinuteLesson`
drives 60 (and, in the prefix-property test, 30x2) turns through the
real `tutor.app.agent_loop.run_turn` and a real `Session`/`PromptLog`,
using a fake LLM whose `stream_chat(...)` and single-argument `emit`
match the real interfaces (the two interface bugs noted in an earlier
draft of this file -- `llm.stream()` vs `stream_chat()`, 2-arg vs 1-arg
`emit` -- are fixed in the test doubles; every original assertion is
kept, none weakened). Measured on this Windows dev machine with
`Budget.scaled(6000)`:

- **Evictions**: 48 of 60 turns' `run_turn` calls carried an
  `eviction_reprefill` event on `TurnResult.events` (small 6000-token
  ceiling forces frequent head-trimming, as intended for a test budget).
- **Max tokens_used**: 1318, comfortably under `ceiling - margin` = 5800.
  **Correction (review pass 2, finding 3):** an earlier draft of this
  note expected `history` to scale to "~4,000" tokens at this 6,000
  ceiling and called 48/60 evictions with max `tokens_used` ~1,318
  surprising. That expectation was simply wrong arithmetic.
  `Budget.scaled()` (`tutor/app/prompt.py`) subtracts the *fixed*
  per-request overhead -- `system(800) + newest(2500) + generation(2000)
  = 5300` -- from the ceiling **first**, and only splits the *remainder*
  80/20 between `history` and `margin`. At a 6,000-token ceiling the
  remainder is only `6000 - 5300 = 700`, so `history` scales to `560`,
  not ~4,000, and `evict()`'s real trigger
  (`operating_ceiling = budget.system + budget.history`) is `800 + 560 =
  1,360` tokens -- not ~4,000. 48/60 evictions with a max post-eviction
  `tokens_used` of 1,318 (just under the real 1,360 threshold) is exactly
  what the code does at this ceiling, not an artefact of a miscounted
  token estimator or of eviction running independent of the numbers. At
  the real production ceiling (`cfg.server.ctx_size = 32768`,
  `config/dev.toml`), `Budget.scaled(32768)` reduces exactly to the
  hand-tuned default (`history = 22000`, `operating_ceiling = 22800`) --
  there is no starvation there; only the small 6K *test* ceiling is this
  aggressive. This small-ceiling test therefore exercises eviction
  *mechanics* (does the head-trim-and-reprefill machinery work at all
  under pressure), not a realistic eviction *frequency* -- a real 32K-ceiling
  lesson will evict far less often than 48/60 turns. See
  `docs/review_2026-09-20_pass2.md` finding 3 for the full numbers at
  every ceiling (6K/8K/16K/32K) and the non-linearity this fixed-overhead-
  first design produces at small ceilings.
- **RSS growth** (turn 10 -> turn 59): ~0.05 MB (effectively flat -- no
  leak pattern over the run).
- **Child process count**: constant throughout (no worker processes
  spawned by this fake-only path).
- **Open file/handle count**: stable within the test's `<= 5` tolerance.
- **Prefix property**: holds on every turn pair that did not evict;
  the sole intentional exception is a turn whose internal eviction
  ran before its model call, exactly as `tutor/app/prompt.py` documents.

These numbers now exercise the actual production `run_turn` path (not a
bypass), so they are a real M5 result for this machine; M6 re-measures
on the target Dell hardware per the plan's note below.

## Live check: prompt-cache hit across two turns (real llama-server)

`tests/test_turn_live.py::test_second_turn_hits_prompt_cache_and_can_refer_to_first_turn`
runs two consecutive turns in one session against the real dev
`llama-server` (`config/dev.toml`) and the libzim fixture registry, over
the real HTTP/SSE route stack (`build_deps` -> `create_app` ->
`TestClient`). It asserts (a) the session's `PromptLog` still contains
turn 1's exact user text after turn 2 (plumbing: turn 2's request was
built by appending to the *same* log, not a fresh one), and (b) turn 2's
reported `cached_tokens` is at least 70% of turn 1's total prompt
tokens. Measured run: turn 1 `tokens_used` = 210, turn 2
`cached_tokens` = 564 (turn 2's prompt reuses essentially all of turn
1's prefix plus its own new content, well over the 70% bar) -- confirming
the byte-prefix property now reaches the server's prompt cache in
production, not just in unit tests. Full `tests/test_turn_live.py` run:
5 passed, 1 xfailed (pre-existing, documented model-behavior flake in
`test_factual_question_pre_retrieves_streams_and_resolves_citations`,
unrelated to this change -- see `docs/M3_report.md`).

## What remains to be MEASURED on real hardware

Not run here, and explicitly out of scope for this WP (no vendor
GPU-tool shell-outs; `gpu_clock_mhz`/`gpu_temp_c` are always `None` in
this WP's wiring):

- A real 30-minute lesson against a live `llama-server` process on the
  target Dell machine: actual RSS growth, thread/handle counts, and
  child-process behavior for the *real* retrieval worker + inference
  host (not fakes), sustained over 30 minutes of real generations.
- Real GPU clock/temperature sampling (via whatever vendor tool ends up
  wired in) to fill `gpu_clock_mhz`/`gpu_temp_c` in `turns.jsonl`.
- Real `first_token_ms` / `tokens_per_second` distributions under actual
  model load, decode-gate thermal behavior mentioned in spec §15's
  acceptance table, and confirmation there is no sustained swap / queue
  growth over a real 30-minute session -- the Windows dev-machine numbers
  above are indicative only, per the plan's WP-C4 note that "resource
  measurements on Windows are indicative only; the numbers that count are
  re-taken on the Dell in M6."
- The `agent_loop.run_turn` contract mismatch is resolved (see above);
  what remains is re-running the same simulated-lesson and live
  prompt-cache checks on the target Dell hardware for the M6 gate.

## Two-stage eviction: uncited evidence dropped before whole turns

Approved change to spec §8.1's eviction order. §8.1 literally says "drop
the oldest `[user, tool, assistant]` triples from the head of the
history until it fits" -- a whole-triple-at-a-time rule with no
sub-triple granularity. The project owner approved a cheaper-first
refinement that stays inside that rule's spirit (still a single head
edit per eviction, still never splices from the middle of a *retained*
triple's meaning) but adds a step before it:

- **Stage 1** (new): walk the oldest turns first; for each evidence tool
  message never cited by any *retained* assistant reply, drop the
  passage TEXT and replace it with a short stub (`"[evidence dropped to
  save space: N passages, never cited]"`), keeping the message list
  valid for the OpenAI wire format. The passage id is forgotten
  (`PromptLog._seen_ids`), so if research returns it again later it is
  re-sent in full under a new label; the old label is never resolvable
  again. Evidence cited by a retained assistant message is never touched
  in this stage.
- **Stage 2** (existing behavior, unchanged): only if stage 1 alone
  didn't free enough room, evict whole oldest `[user, tool*, assistant]`
  turns, exactly as before (including re-protecting evidence still cited
  by a surviving assistant message).

Both stages happen inside one `PromptLog.evict()` call and are reported
as a single `EvictionEvent`, still exactly one byte-prefix break per
eviction (`tests/test_prompt_builder.py::test_stage1_keeps_log_valid_and_prefix_breaks_exactly_once`
and the pre-existing `test_evict_breaks_prefix_property_exactly_once_then_holds_again`
both stay green). `EvictionEvent` gained `dropped_uncited_passages` and
`dropped_uncited_tokens`; both are forwarded through the `eviction_reprefill`
event dict, the SSE `eviction` payload, and `/api/status.last_eviction`
(`tutor/app/agent_loop.py`, `tutor/app/compose.py`).

**Numbers from the simulated 60-turn lesson**
(`tests/test_resource_discipline.py::TestSimulatedThirtyMinuteLesson::test_sixty_turns_stay_within_ceiling_with_bounded_rss_growth`,
6K ceiling): 48 `eviction_reprefill` events fire over 60 turns, and in
this specific fixture every one of them is stage-2-only (0 evictions
satisfied by stage 1 alone) -- because the fixture's fake LLM answer
cites `[S1]` on every single turn, so every turn's own evidence is
"cited by a retained assistant message" right up until that turn itself
is evicted. This is the expected, unsurprising case: stage 1 only pays
off when a lesson accumulates evidence the model never ends up citing.
To confirm stage 1 actually does its job, a synthetic variant of the
same 60-turn loop where the fake answer cites nothing (`docs/M5_notes.md`
scratch check, not committed as a test since it duplicates the fixture)
shows the two-stage split clearly: of 48 eviction events, 29 are
satisfied by stage 1 alone (no whole turn evicted, only stale evidence
text stubbed), 19 need stage 2 as well, and 60 passages' worth of
never-cited evidence text is dropped in total. The dedicated unit tests
in `tests/test_prompt_builder.py`
(`test_stage1_alone_stubs_uncited_evidence_oldest_first_no_turn_evicted`,
`test_stage1_never_drops_evidence_cited_by_a_retained_assistant_message`,
`test_stage2_still_fires_when_stage1_is_not_enough`,
`test_stubbed_passage_is_resent_in_full_under_a_new_label_on_re_retrieval`)
exercise this split directly with a fixture where evidence is never
cited, isolating stage 1 from stage 2.

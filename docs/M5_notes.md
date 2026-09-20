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

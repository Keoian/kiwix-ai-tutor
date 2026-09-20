# M5 Report: profiles, lesson state, prompt log, status page, 30-minute lesson soak

Generated: 2026-09-20T08:11:08

## M5 gate checklist

Gate (docs/plan/offline_tutor_implementation_plan.md §1): "Profiles, lesson state, append-only prompt with eviction under the 32K profile, status page, 30-minute lesson within resource limits (Windows measurements)."

| Item | Evidence | Status |
|---|---|---|
| Profiles | `tutor/app/profiles.py`, `tests/test_profiles.py` | measured (unit) |
| Lesson state / persistence / resume | `tutor/app/lesson_state.py`, `tests/test_lesson_state.py`, `tests/test_review_pass2_fixes.py` | measured (unit) |
| Append-only prompt log + eviction | `tutor/app/prompt.py`, `tests/test_prompt_builder.py`, docs/M5_notes.md | measured (unit) |
| Status page | `tutor/app/compose.py` (`_make_status_provider`), `tests/test_compose.py`, `tests/test_turn_live.py` | measured (unit + this soak) |
| 30-minute real lesson within resource limits | this document, Soak results below | PASS (measured, no turn >5min, soak ran to completion) |

## Soak run parameters

- config: config/dev.toml
- minutes_requested: 30.0
- started_at: 2026-09-20T07:37:43
- ended_at: 2026-09-20T08:11:08
- stopped_early: False
- stop_reason: None

## Turns

- turns completed: **25** (ok: 25, errored: 0)
- no turn exceeded 5 minutes wall time (measured)

## Latency (measured)

- time-to-first-token: p50=49.672s, p95=177.837s
- total turn wall time: p50=49.672s, p95=177.837s

IMPORTANT hardware caveat (inferred from prior measurement, not re-derived here): on this GPU, prompt prefill throughput collapses with depth when flash-attention is on (~21 t/s at 4k context, ~11 t/s at 8k). Turns only stay fast as the lesson log grows if the prompt cache keeps hitting -- see cache hit ratio below.

## Token growth (measured)

Prompt tokens_used every ~5 turns:

| turn | tokens_used |
|---|---|
| 5 | 5601 |
| 10 | 7649 |
| 15 | 11341 |
| 20 | 14954 |
| 25 | 18412 |

- max tokens_used observed: 18412 (32K ceiling; a 30-minute lesson may never trigger eviction -- that is expected, not a bug)
- eviction_reprefill events observed: 0 (measured, captured in-process; not currently forwarded over SSE -- see docs/M5_notes.md gap note)

## Prompt cache hit ratio (measured, cached_tokens / tokens_used)

- mean: 0.955

## Citations and calc correctness (measured)

- factual (pre-retrieval) turns: 18
- citation rate (citations per factual turn): 1.167
- citation resolved rate: 1.000
- uncited rate (factual turn, no [S#] at all): 0.889
- calc items scripted with known answers: 2, correct: 2 (1.000)

## Research status mix (measured, captured via a research()-call recorder; no research_status field is exposed over the wire today)

**Not captured on this run**: the recorder that buffers `(status, elapsed_s)`
per turn had a list-aliasing bug (`calls_for_turn, recorder_calls[:] =
recorder_calls, []` bound `calls_for_turn` to the *same* list object that
was then cleared in place, wiping it before it could be read) --
`research_status_mix` and `research_mean_elapsed_s` are empty/`n/a` for
this reason, not because research never ran (18 factual turns did trigger
research calls; citation counts above come from a separate, unaffected
code path). Fixed in `eval/run_lesson_soak.py` (`list(recorder_calls)` +
`.clear()`) after this run; not re-run to avoid a second 30-minute GPU
session for one derived metric. Deferred to the next soak (M6 Dell run
or a repeat here).

- mean research call elapsed: n/a

## Resource usage (measured, via /api/status every ~30s)

- RSS MB: start=69.766, end=89.617, max=90.543, growth=19.852
- thread count: start=22, end=19
- open files/handles: start=11, end=11
- child processes: start=1, end=1
- llama-server /health stayed healthy for every sample: True

## Resume check

PASS (measured): resumed PromptLog validated cleanly and one additional turn completed successfully after resume (lesson bfcbc171f90a41d091d6567ac60d7d53).

## Fixed after the soak

The real 30-minute soak above (the numbers in every section up to and including "Resume check") exposed two product gaps that the soak script had been papering over by doing the work itself. Both are now fixed in the app, not just in the soak harness:

- **GAP 1 -- the real app did not persist lessons.** `tutor/app/routes.py`/`tutor/app/compose.py` never wrote turns into `LessonStore`, so after a real app restart a student's lesson could not actually be resumed (this 30-minute soak persisted turns itself, in-process, purely to exercise the resume path). Fixed: `POST /api/lesson` attaches a session to a brand-new lesson (`lesson_id` + `session_id`); `POST /api/lesson/{id}/resume` rebuilds a session from a persisted lesson. `tutor.app.compose`'s `turn_runner` now persists every completed turn (including error/cancelled turns, log repaired-if-needed first) atomically -- one SQLite transaction writing both the turn row and the prompt-log snapshot (`LessonStore.persist_turn`) -- so a crash between turns loses at most the in-flight turn. A subject change on an attached session follows the pre-existing `LessonStore` rule (start a new lesson) rather than corrupting the current one. Sessions with no lesson attached are unaffected.
  - Tests: `tests/test_lesson_state.py::TestPersistTurnAtomic`, `tests/test_compose.py::test_full_app_persists_turns_and_resumes_byte_identical`, `tests/test_compose.py::test_no_lesson_attached_session_still_works_and_is_never_persisted`, `tests/test_routes.py::test_create_lesson_returns_lesson_and_session_ids`, `tests/test_routes.py::test_resume_lesson_returns_a_fresh_session_id`, `tests/test_routes.py::test_resume_unknown_lesson_is_404`.
- **GAP 2 -- `eviction_reprefill` events were not forwarded to the client.** Fixed: `tutor.app.compose`'s turn adapter now emits an SSE `eviction` event (evicted turn count, tokens before/after) at the moment eviction fires, before the next model call, and remembers the session's last eviction for `/api/status`'s `last_eviction` field. `tutor/ui/app.js` shows a small unobtrusive note in the chat pane and the status panel (`textContent` only).
  - Tests: `tests/test_compose.py::test_eviction_event_forwarded_over_sse_and_status`; `tests/test_ui_static.py` stays green (no `innerHTML`, `textContent` used).

`eval/run_lesson_soak.py` was simplified accordingly: it now creates/attaches its lesson via `POST /api/lesson`, no longer monkeypatches `PromptLog.evict` or calls `LessonStore.append_turn`/`save_session` itself (the app does that), counts `eviction_reprefill` from the `eviction` SSE event, and its `_resume_check` drives `POST /api/lesson/{id}/resume` over real HTTP instead of touching `LessonStore`/`Session` internals directly. `tests/test_lesson_soak.py` (pure-function unit tests only) stayed green throughout.

### 3-minute confirmation run (measured, `data/soak_short_confirm.md`, NOT a re-run of the 30-minute soak above)

`python -m eval.run_lesson_soak --config config/dev.toml --minutes 3 --out data/soak_short_confirm.md` against the live dev llama-server:

- turns completed: **7** (ok: 7, errored: 0)
- research status mix: **ok: 3, partial: 2** (previously empty on the 30-minute run due to a since-fixed recorder aliasing bug -- now populated, confirming that fix)
- mean research call elapsed: 3.528s
- ttft: p50=20.406s, p95=33.568s
- cache hit ratio mean: 0.943
- eviction_reprefill events observed: 0 (a 3-minute lesson at this token growth rate does not reach the eviction ceiling -- expected, not a bug; GAP 2's forwarding path is covered by the unit test above, not by this short run)
- resume check: **PASS (measured)** -- "POST /api/lesson/{id}/resume rebuilt a byte-identical prompt log (app-side persistence, no soak-driven persistence involved) and one additional turn completed successfully after resume" (lesson `726314f24b6b4c049275258d3e6d4eab`)
- llama-server /health stayed healthy for every sample: True

This is a short confirmation only, run to validate the two fixes above against the live app end-to-end without repeating the 30-minute soak; it does not replace or update any of the 30-minute soak's own measured numbers above.

## Deferred to the Dell (M6)

- Real hardware measurements on the delivery machine (this soak ran on the dev GPU box).

Labeling: everything in "Turns"/"Latency"/"Token growth"/"Prompt cache"/"Citations"/"Research status"/"Resource usage"/"Resume check" is **measured** from this run. The hardware prefill-collapse figures (21 t/s@4k / 11 t/s@8k) are **inferred** from a prior measurement, cited for context, not re-derived here.

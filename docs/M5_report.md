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

## Deferred to the Dell (M6)

- Real hardware measurements on the delivery machine (this soak ran on the dev GPU box).
- Automatic wiring of session turns into `LessonStore.append_turn`/`save_session` from `tutor.app.compose._make_turn_runner` (this soak drives that persistence itself, in-process, to exercise the resume path -- routes.py does not yet do this on every turn).
- Forwarding `eviction_reprefill` over the SSE wire (currently dropped by `compose._make_turn_runner`'s adapter; this soak reads eviction events in-process instead).

Labeling: everything in "Turns"/"Latency"/"Token growth"/"Prompt cache"/"Citations"/"Research status"/"Resource usage"/"Resume check" is **measured** from this run. The hardware prefill-collapse figures (21 t/s@4k / 11 t/s@8k) are **inferred** from a prior measurement, cited for context, not re-derived here.

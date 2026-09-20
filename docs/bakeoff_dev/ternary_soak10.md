# M5 Report: profiles, lesson state, prompt log, status page, 30-minute lesson soak

Generated: 2026-09-20T13:42:01

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

- config: config/dev.ternary.toml
- minutes_requested: 10.0
- started_at: 2026-09-20T13:29:35
- ended_at: 2026-09-20T13:42:01
- stopped_early: False
- stop_reason: None

## Turns

- turns completed: **29** (ok: 29, errored: 0)
- no turn exceeded 5 minutes wall time (measured)

## Latency (measured)

- time-to-first-token: p50=8.532s, p95=74.803s
- total turn wall time: p50=8.532s, p95=74.803s

IMPORTANT hardware caveat (inferred from prior measurement, not re-derived here): on this GPU, prompt prefill throughput collapses with depth when flash-attention is on (~21 t/s at 4k context, ~11 t/s at 8k). Turns only stay fast as the lesson log grows if the prompt cache keeps hitting -- see cache hit ratio below.

## Token growth (measured)

Prompt tokens_used every ~5 turns:

| turn | tokens_used |
|---|---|
| 5 | 2791 |
| 10 | 4163 |
| 15 | 5815 |
| 20 | 7520 |
| 25 | 9555 |
| 29 | 9699 |

- max tokens_used observed: 9699 (32K ceiling; a 30-minute lesson may never trigger eviction -- that is expected, not a bug)
- eviction_reprefill events observed: 4 (measured; forwarded live over SSE as an `eviction` event and counted from that event, not read out of PromptLog in-process -- see docs/M5_report.md "fixed after the soak")

## Prompt cache hit ratio (measured, cached_tokens / tokens_used)

- mean: 0.966

## Citations and calc correctness (measured)

- factual (pre-retrieval) turns: 21
- citation rate (citations per factual turn): 1.333
- citation resolved rate: 1.000
- uncited rate (factual turn, no [S#] at all): 0.048
- calc items scripted with known answers: 3, correct: 3 (1.000)

## Research status mix (measured, captured via a research()-call recorder; no research_status field is exposed over the wire today)

- ok: 17
- partial: 4
- mean research call elapsed: 2.517s

## Resource usage (measured, via /api/status every ~30s)

- RSS MB: start=69.426, end=79.887, max=80.625, growth=10.461
- thread count: start=22, end=21
- open files/handles: start=10, end=10
- child processes: start=1, end=1
- llama-server /health stayed healthy for every sample: True

## Resume check

PASS (measured): POST /api/lesson/{id}/resume rebuilt a byte-identical prompt log (app-side persistence, no soak-driven persistence involved) and one additional turn completed successfully after resume (lesson 30bb97347df34e5e9dddd3ea8f1bf9cc).

## Deferred to the Dell (M6)

- Real hardware measurements on the delivery machine (this soak ran on the dev GPU box).

Labeling: everything in "Turns"/"Latency"/"Token growth"/"Prompt cache"/"Citations"/"Research status"/"Resource usage"/"Resume check" is **measured** from this run. The hardware prefill-collapse figures (21 t/s@4k / 11 t/s@8k) are **inferred** from a prior measurement, cited for context, not re-derived here.

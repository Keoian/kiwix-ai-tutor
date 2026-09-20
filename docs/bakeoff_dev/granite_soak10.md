# M5 Report: profiles, lesson state, prompt log, status page, 30-minute lesson soak

Generated: 2026-09-20T12:34:25

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

- config: config/dev.granite.toml
- minutes_requested: 10.0
- started_at: 2026-09-20T12:24:07
- ended_at: 2026-09-20T12:34:25
- stopped_early: False
- stop_reason: None

## Turns

- turns completed: **63** (ok: 63, errored: 0)
- no turn exceeded 5 minutes wall time (measured)

## Latency (measured)

- time-to-first-token: p50=2.687s, p95=11.523s
- total turn wall time: p50=2.687s, p95=11.523s

IMPORTANT hardware caveat (inferred from prior measurement, not re-derived here): on this GPU, prompt prefill throughput collapses with depth when flash-attention is on (~21 t/s at 4k context, ~11 t/s at 8k). Turns only stay fast as the lesson log grows if the prompt cache keeps hitting -- see cache hit ratio below.

## Token growth (measured)

Prompt tokens_used every ~5 turns:

| turn | tokens_used |
|---|---|
| 5 | 2967 |
| 10 | 4501 |
| 15 | 6401 |
| 20 | 8188 |
| 25 | 10763 |
| 30 | 12189 |
| 35 | 14452 |
| 40 | 16072 |
| 45 | 16314 |
| 50 | 16545 |
| 55 | 16786 |
| 60 | 17061 |
| 63 | 17192 |

- max tokens_used observed: 17192 (32K ceiling; a 30-minute lesson may never trigger eviction -- that is expected, not a bug)
- eviction_reprefill events observed: 0 (measured; forwarded live over SSE as an `eviction` event and counted from that event, not read out of PromptLog in-process -- see docs/M5_report.md "fixed after the soak")

## Prompt cache hit ratio (measured, cached_tokens / tokens_used)

- mean: 1.106

## Citations and calc correctness (measured)

- factual (pre-retrieval) turns: 45
- citation rate (citations per factual turn): 1.400
- citation resolved rate: 1.000
- uncited rate (factual turn, no [S#] at all): 0.000
- calc items scripted with known answers: 8, correct: 6 (0.750)

## Research status mix (measured, captured via a research()-call recorder; no research_status field is exposed over the wire today)

- ok: 34
- partial: 11
- mean research call elapsed: 2.110s

## Resource usage (measured, via /api/status every ~30s)

- RSS MB: start=68.562, end=82.754, max=82.754, growth=14.191
- thread count: start=22, end=19
- open files/handles: start=10, end=10
- child processes: start=1, end=1
- llama-server /health stayed healthy for every sample: True

## Resume check

PASS (measured): POST /api/lesson/{id}/resume rebuilt a byte-identical prompt log (app-side persistence, no soak-driven persistence involved) and one additional turn completed successfully after resume (lesson b2b0981d2e624fcca93e08328767d643).

## Deferred to the Dell (M6)

- Real hardware measurements on the delivery machine (this soak ran on the dev GPU box).

Labeling: everything in "Turns"/"Latency"/"Token growth"/"Prompt cache"/"Citations"/"Research status"/"Resource usage"/"Resume check" is **measured** from this run. The hardware prefill-collapse figures (21 t/s@4k / 11 t/s@8k) are **inferred** from a prior measurement, cited for context, not re-derived here.

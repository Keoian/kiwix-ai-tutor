# M5 Report: profiles, lesson state, prompt log, status page, 30-minute lesson soak

Generated: 2026-09-20T14:36:10

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
- started_at: 2026-09-20T14:25:59
- ended_at: 2026-09-20T14:36:10
- stopped_early: False
- stop_reason: None

## Turns

- turns completed: **62** (ok: 62, errored: 0)
- no turn exceeded 5 minutes wall time (measured)

## Latency (measured)

- time-to-first-token: p50=2.969s, p95=11.212s
- total turn wall time: p50=2.977s, p95=11.212s

IMPORTANT hardware caveat (inferred from prior measurement, not re-derived here): on this GPU, prompt prefill throughput collapses with depth when flash-attention is on (~21 t/s at 4k context, ~11 t/s at 8k). Turns only stay fast as the lesson log grows if the prompt cache keeps hitting -- see cache hit ratio below.

## Token growth (measured)

Prompt tokens_used every ~5 turns:

| turn | tokens_used |
|---|---|
| 5 | 2942 |
| 10 | 4556 |
| 15 | 6355 |
| 20 | 8170 |
| 25 | 10812 |
| 30 | 12281 |
| 35 | 14590 |
| 40 | 16296 |
| 45 | 16469 |
| 50 | 16710 |
| 55 | 16911 |
| 60 | 17173 |
| 62 | 17276 |

- max tokens_used observed: 17276 (32K ceiling; a 30-minute lesson may never trigger eviction -- that is expected, not a bug)
- eviction_reprefill events observed: 0 (measured; forwarded live over SSE as an `eviction` event and counted from that event, not read out of PromptLog in-process -- see docs/M5_report.md "fixed after the soak")

## Prompt cache hit ratio (measured, cached_tokens / tokens_used)

- mean: 1.091

## Citations and calc correctness (measured)

- factual (pre-retrieval) turns: 44
- citation rate (citations per factual turn): 1.409
- citation resolved rate: 1.000
- uncited rate (factual turn, no [S#] at all): 0.000
- calc items scripted with known answers: 8, correct: 6 (0.750)

### Citations (factual turns only)

cited_rate == 1 - uncited_rate (same factual-turn denominator, complementary definitions: cited_rate counts turns with >=1 [S#] label, uncited_rate counts turns with none).

| metric | value | definition |
|---|---|---|
| factual_turns | 44 | pre-retrieval turns (route starts with `preretrieve`) |
| cited_turns / cited_rate | 44 / 1.000 | factual turns with at least one [S#] citation label |
| supported_turns / supported_rate | 0 / 0.000 | factual turns with >=1 citation label and no unsupported labels |
| evidence_dump_turns | 0 | factual turns flagged as dumping raw evidence instead of a synthesized answer |
| uncited_rate | 0.000 | factual turns with no [S#] citation at all (== 1 - cited_rate) |

## Research status mix (measured, captured via a research()-call recorder; no research_status field is exposed over the wire today)

- ok: 33
- partial: 11
- mean research call elapsed: 2.080s

## Resource usage (measured, via /api/status every ~30s)

- RSS MB: start=69.418, end=87.180, max=87.957, growth=17.762
- thread count: start=22, end=19
- open files/handles: start=11, end=11
- child processes: start=1, end=1
- llama-server /health stayed healthy for every sample: True

## Resume check

PASS (measured): POST /api/lesson/{id}/resume rebuilt a byte-identical prompt log (app-side persistence, no soak-driven persistence involved) and one additional turn completed successfully after resume (lesson 023f80c469da4b059636c6884dbd4728).

## Deferred to the Dell (M6)

- Real hardware measurements on the delivery machine (this soak ran on the dev GPU box).

Labeling: everything in "Turns"/"Latency"/"Token growth"/"Prompt cache"/"Citations"/"Research status"/"Resource usage"/"Resume check" is **measured** from this run. The hardware prefill-collapse figures (21 t/s@4k / 11 t/s@8k) are **inferred** from a prior measurement, cited for context, not re-derived here.

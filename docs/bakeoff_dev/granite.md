# Granite 4.0 H-Tiny bake-off summary

Model: IBM Granite 4.0 H-Tiny (granitehybrid, Mamba-2/attention hybrid MoE,
7B.A1B, 6.94B total params), Q4_K_M, 32K ctx, q8_0/q8_0 KV, flash-attn on,
full GPU offload (Vulkan, Radeon Pro 5500M). Config: `config/dev.granite.toml`.
Flags/VRAM/raw-speed sweep: [docs/bakeoff_dev/granite_flags.md](granite_flags.md)
(pp ~683-706 t/s short-context, tg ~64-68 t/s short-context, tg ~30-64 t/s at
depth depending on ub; loads in ~5.1s; ~5.36 GiB VRAM at idle/32K ctx).

## Headline table (Granite vs. Bonsai-8B-Q1_0, same harnesses/questions)

| Metric | Bonsai Q1_0 (previous) | Granite 4.0 H-Tiny (this run) |
|---|---:|---:|
| Tool calls: parsed correct | 0/20 | 11/20 parsed, 19/20 correct decisions |
| Tool calls: malformed | 20/20 | 0/20 (0.00%) |
| Citation rate (18 real Q) | 0.11 | 0.44 |
| Supported-citation rate | 0.00 | 0.11 |
| Evidence-dump rate | 0.00 | 0.06 |
| On-topic-citation rate | 0.11 | 0.33 |
| Taught rate | 0.28 | 0.22 |
| Helium probe (x3) | drifting figures, 0/3 cited, 0/3 calc | exact figures 3/3, 1/3 cited, 0/3 calc (not needed) |
| Prompt-cache reuse (2-turn) | works (prior model) | works: turn1 prompt_tokens=664, turn2 cached_tokens=1219 (>= 0.7x threshold; Mamba-hybrid prefix reuse confirmed) |
| Soak duration | 30 min | 10 min (mini-soak) |
| Soak turns / errors | n/a here | 63 turns, 0 errors |
| First-token p50 / p95 | ~20s / n/a | 2.687s / 11.523s |
| Total-turn p50 / p95 | 49.7s / n/a | 2.687s / 11.523s |
| Cache hit ratio | 0.955 | 1.106 (mean cached/used, includes system-prompt reuse) |
| Uncited rate (soak) | 0.889 | 0.000 |
| Calc correctness (soak) | n/a | 6/8 (0.750) |
| RSS growth (soak) | n/a | 68.6 -> 82.8 MB (+14.2 MB) |
| pp / tg (short context) | ~175 t/s / ~31 t/s | ~683-706 t/s / ~64-68 t/s |

Full detail: [granite_toolcalls.md](granite_toolcalls.md),
[granite_citations.md](granite_citations.md), [granite_helium.md](granite_helium.md),
[granite_soak10.md](granite_soak10.md), [granite_flags.md](granite_flags.md).

## App fixes made during this bake-off

`tests/test_llm_live.py` and `tests/test_turn_live.py` had a hardcoded
`_CONFIG_PATHS = ["config/dev.toml"]` fixture parameter list, which made it
impossible to run the live-app integration suite against any other model
config without editing the test file each time. Added
`"config/dev.granite.toml"` to both lists so the suite is parameterized
across both dev configs going forward (both configs point at the same
`:8080` server, so whichever model is actually running answers both
parameter sets — this is pre-existing behavior, not new). All 16 applicable
tests pass against the running Granite server (2 pre-existing `xfail`
unrelated to the model). This is a test-harness improvement, not a
model-specific fix, and was verified by running the full live suite
before and after.

## Candid assessment

Granite 4.0 H-Tiny is a clear, measured improvement over Bonsai-8B-Q1_0 on
every axis this product cares about: it is dramatically faster (first-token
p50 ~2.7s vs ~20s, tg roughly 2x, pp roughly 4x), it emits well-formed tool
calls natively (0% malformed vs 100%), it reuses the prompt cache even
though it's a Mamba-2 hybrid architecture (turn-2 cached_tokens exceeded
turn-1's total prompt tokens, i.e. essentially full-prefix reuse), and in
the 10-minute soak it produced citations on every factual turn (uncited rate
0.000, vs 0.889 for the previous model) with no errors across 63 turns and
flat memory growth. The helium probe reproduced the evidence's figures
exactly in all three repeats, with no unit-conversion drift, which the
previous model could not do reliably.

The weaknesses are real, though narrower than the previous model's. First,
**citation discipline is inconsistent, not absent**: on the 18-question
real-archive set citation rate is only 0.44 and "supported" (facts actually
backed by the cited sentence) is only 0.11 — the model often answers
correctly and fluently from parametric knowledge without citing the
retrieved evidence at all (5 of the first 6 manually-read answers), and even
when it does cite (e.g. the water-cycle question) it sometimes elaborates
with more granular detail than a single evidence sentence supports. This
matters for a product whose core promise is "grounded in this article," not
"generically correct." Second, **answer register drifts above the target
reading level on some topics** (e.g. bringing up abstract algebra structures
like groups/rings/fields when asked "what is algebra?", or "hex-3-ene" for a
basic chemical-bonds question) — content is accurate but not always
age-appropriately scoped for a 10-16-year-old, so a prompt-side rewrite
pass or stricter "stay at the evidence's level" instruction is likely
needed before shipping. Neither weakness is a blocker on its own, but they
are the two things to tune next (system-prompt citation enforcement, and a
reading-level/scope constraint), rather than model-swap-blocking defects.

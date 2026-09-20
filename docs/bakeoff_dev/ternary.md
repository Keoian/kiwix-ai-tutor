# Ternary-Bonsai-8B bake-off summary

Model: Ternary-Bonsai-8B (dense 8B transformer, 36 layers / 8 KV heads,
ternary/1-bit-family quantization), Q2_0_g64 GGUF, 16K ctx (not 32K --
KV cache is ~72 KiB/token at f16, much larger per-token than Granite's
Mamba-2 hybrid, so this bake-off runs a smaller context ceiling), f16/f16
KV, flash-attn off, ub 128, full GPU offload (Vulkan, Radeon Pro 5500M).
Config: `config/dev.ternary.toml`. Flags/VRAM/raw-speed sweep:
[docs/bakeoff_dev/ternary_flags.md](ternary_flags.md) (pp ~157-229 t/s
short-context, tg ~45-51 t/s short-context, tg ~11-32 t/s at depth
depending on fa/ctk/ub).

## Headline table (Bonsai Q1_0 vs. Granite 4.0 H-Tiny vs. Ternary-Bonsai-8B, same harnesses/questions)

| Metric | Bonsai Q1_0 (previous) | Granite 4.0 H-Tiny | Ternary-Bonsai-8B (this run) |
|---|---:|---:|---:|
| Tool calls: parsed correct | 10/20 parsed, 20/20 correct decisions | 11/20 parsed, 19/20 correct decisions | 10/20 parsed, 20/20 correct decisions |
| Tool calls: malformed | 0/20 (0.00%) | 0/20 (0.00%) | 0/20 (0.00%) |
| Citation rate (18 real Q) | 0.11 | 0.44 | 0.17 |
| Supported-citation rate | 0.00 | 0.11 | 0.00 |
| Evidence-dump rate | 0.00 | 0.06 | 0.00 |
| On-topic-citation rate | 0.11 | 0.33 | 0.17 |
| Taught rate | 0.28 | 0.22 | 0.11 |
| Helium probe (x3) | drifting figures, 0/3 cited, 0/3 calc | exact figures 3/3, 1/3 cited, 0/3 calc | exact figures 3/3, 1/3 cited, 0/3 calc |
| Prompt-cache reuse (2-turn) | works | works: turn1=664, turn2 cached=1219 | works: turn1 prompt_tokens=628, turn2 cached_tokens=1071 (1.7x, well over 0.7x threshold) |
| Soak duration | 30 min | 10 min (mini-soak) | 10 min (mini-soak) |
| Soak turns / errors | n/a | 63 turns, 0 errors | 29 turns, 0 errors |
| First-token p50 / p95 | ~20s / n/a | 2.687s / 11.523s | 8.532s / 74.803s |
| Total-turn p50 / p95 | 49.7s / n/a | 2.687s / 11.523s | 8.532s / 74.803s |
| Cache hit ratio | 0.955 | 1.106 | 0.966 |
| Uncited rate (soak) | 0.889 | 0.000 | 0.048 |
| Calc correctness (soak) | n/a | 6/8 (0.750) | 3/3 (1.000) |
| Eviction events (soak) | n/a | 0 (never hit ceiling in 32K) | 4 (16K ceiling hit and evicted correctly) |
| RSS growth (soak) | n/a | 68.6 -> 82.8 MB (+14.2 MB) | 69.4 -> 79.9 MB (+10.5 MB) |
| pp / tg (short context) | ~175 t/s / ~31 t/s | ~683-706 t/s / ~64-68 t/s | ~157-229 t/s / ~45-51 t/s |

Full detail: [ternary_toolcalls.md](ternary_toolcalls.md),
[ternary_citations.md](ternary_citations.md), [ternary_helium.md](ternary_helium.md),
[ternary_soak10.md](ternary_soak10.md), [ternary_flags.md](ternary_flags.md).

## App fixes made during this bake-off

Same pre-existing test-harness limitation as the Granite run: `_CONFIG_PATHS`
in `tests/test_llm_live.py` and `tests/test_turn_live.py` was a hardcoded
list. Added `"config/dev.ternary.toml"` to both, alongside the existing
`dev.toml`/`dev.granite.toml` entries, so the live-app integration suite is
now parameterized across all three dev configs. 24 passed / 3 xfailed
(pre-existing, unrelated to the model) against the running Ternary server;
no other app bug found or fixed this round -- the `Budget.for_ceiling`
plumbing already reads `cfg.server.ctx_size` (confirmed in
`tutor/app/compose.py` line 509 and `tutor/app/prompt.py`), so the 16K
ceiling drove real eviction in the soak rather than an overflow, exactly as
expected without any code change.

## Candid assessment

Ternary-Bonsai-8B is a real step up from the previous Bonsai-8B-Q1_0
build on **numeric precision** (exact,
byte-for-byte helium figures in 3/3 runs, same as Granite, vs Q1_0's
drifting digits). It also reuses the prompt cache correctly across turns
(a >1.7x cache hit on turn 2 vs turn 1's total, well past the 0.7x bar) and
completed a real 10-minute mini-soak with zero errors and flat memory
growth (+10.5 MB), and it is the only one of the three models in this
series to actually exercise -- and pass -- prompt-log eviction, since its
larger dense-transformer KV cache forced a 16K context ceiling that the
soak's lesson log grew past four times, each time evicting cleanly with no
overflow and no errored turn.

The weaknesses are more serious than Granite's, though narrower than
Q1_0's. First, **latency is materially worse than Granite and, at the p95
tail, worse than even the previous model**: first-token p50 is 8.5s (vs
Granite's 2.7s) and p95 balloons to 74.8s in the 10-minute soak -- almost
certainly the flash-attention-off, f16 KV, 36-layer dense-attention
prefill cost compounding as the lesson log grows, the same prefill-collapse
pattern noted for this GPU in the Granite report, just worse here because
Ternary can't use flash-attn (off by config) and has 8x the KV-cache
footprint per token of Granite's Mamba hybrid. This is the single biggest
practical usability problem: a 75-second worst-case wait mid-lesson is a
hard sell for a 10-16-year-old's attention span. Second, **citation
discipline actually regressed on the real-archive set**: citation rate
0.17 (vs Granite's 0.44), supported-citation rate 0.00 (vs Granite's
already-weak 0.11), and all 6 manually-read answers in the citations dump
were completely uncited -- fluent, factually reasonable, but pure
parametric-knowledge answers with no `[S#]` tag at all, and reading level
skews slightly high for the target age (LaTeX equations, "coordination
bond"-style jargon) same as Granite's pattern but with zero grounding
gesture even on the water-cycle-style multi-step questions. In the soak,
by contrast, citation discipline was fine (uncited rate 0.048, calc 3/3) --
so the model *can* cite reliably when the turn-runner's retrieval-first
routing kicks in; the standalone citations-eval harness's route ("preretrieve")
apparently produces weaker grounding pressure than the live app's actual
turn flow. Net: Ternary is the safer of the two upgrade candidates on tool
calling and numeric correctness, but Granite remains faster and more
consistently grounded on the citation axis; if p95 latency in a real
lesson can't be brought down (e.g. by revisiting the flash-attn-off
decision or shrinking ub further), Granite is the better default and
Ternary is a fallback for its stronger tool-call and eviction behavior.

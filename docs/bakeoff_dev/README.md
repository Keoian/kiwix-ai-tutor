# Dev bake-off: Bonsai Q1_0 vs. Ternary Bonsai Q2_0_g64 vs. Granite H-Tiny Q4_K_M

**Metric note.** The 10-minute soak harness (`eval/run_lesson_soak.py`,
commit 473d354 onward) reports two related but different citation metrics
over factual (pre-retrieval) turns: `uncited`/`cited_rate` is a mechanical
presence check (does the answer contain at least one `[S#]` label
anywhere), while `supported_rate` is a stricter mechanical word-overlap
check (`tutor.app.citations.is_supported`/`resolve_citations`) that only
counts a turn as `supported` if the sentence carrying the label shares
enough content terms with the passage it cites — a turn can be 100% cited
and 0% supported if every citation is a single tag stapled to the end of a
multi-sentence paragraph rather than attached to the specific sentence it
backs. Older soak docs in this repo (`ternary_soak10.md`, `granite_soak10.md`,
`M5_report.md`) predate the `supported` metric and only report
presence-only `cited`/`uncited`.

## Comparison table

Sources: [granite.md](granite.md), [granite_flags.md](granite_flags.md),
[granite_citations.md](granite_citations.md), [granite_helium.md](granite_helium.md),
[granite_soak10.md](granite_soak10.md), [ternary.md](ternary.md),
[ternary_flags.md](ternary_flags.md), [ternary_citations.md](ternary_citations.md),
[ternary_soak10.md](ternary_soak10.md),
[../citation_experiment.md](../citation_experiment.md),
[../dev_runtime.md](../dev_runtime.md), [../M5_report.md](../M5_report.md),
[../toolcall_verification.md](../toolcall_verification.md); this run's new
files: [q1_0_soak10.md](q1_0_soak10.md), [granite_soak10_v2.md](granite_soak10_v2.md).

| Row | Bonsai Q1_0 | Ternary Bonsai Q2_0_g64 | Granite H-Tiny Q4_K_M |
|---|---|---|---|
| Loads on this Vulkan build | Yes (dev_runtime.md) | Yes, after switching to the `_g64` GGUF — the legacy Q2_0 file (group-128 packing) did **not** load; PQ2_0 and Q2_0_g64 did (ternary_flags.md) | Yes, first try, no CPU fallback needed (granite_flags.md) |
| Context fitting <= 5.6 GiB | 32768 ctx fits (dev_runtime.md) | 16384 ctx (cut from a 32K/20K target — f16 KV, no flash-attn, ~72 KiB/token, would not fit at 32K) (ternary_flags.md) | 32768 ctx fits (granite_flags.md) |
| VRAM | 3,584 MiB @ 32K, q8_0/q8_0 KV (dev_runtime.md) | ~5.54 GiB @ 16K, f16/f16 KV (thin margin, ~69 MiB headroom under the 5.6 GiB cap) (ternary_flags.md) | ~5.36 GiB @ 32K, q8_0/q8_0 KV (granite_flags.md) |
| pp t/s d0/4k/8k | ~175 t/s short-context only; no d4k/d8k pp figure measured for Q1_0 itself — only the hardware-level fa-trap reference (21 t/s@4k / 11 t/s@8k with fa on) is on record (dev_runtime.md) *(not like-for-like: different measurement granularity)* | 157-229 / ~174.5 (noisy) / 122.2 t/s (ternary_flags.md) | 683-706 / ~640-679 / 640.4 t/s (fa=0,f16,ub512) (granite_flags.md) |
| tg t/s d0/4k/8k | ~31.9 t/s short-context only; no per-depth tg breakdown on record (dev_runtime.md) *(not like-for-like)* | 50.7 / 26.7 / 10.6 t/s (ternary_flags.md) | 67.8-68.1 / 63.9 / 30.6 t/s (granite_flags.md) |
| Tool calls: malformed / correct decisions | Conflicting sources in this repo: `toolcall_verification.md` (2026-09-19, `eval/toolcall_harness`, n=20) reports **0/20 malformed, 20/20 correct decisions**; `ternary.md`'s headline table instead lists Q1_0 as "20/20 malformed, 0/20 parsed correct" for the same "previous model" row. Not reconciled here — flagging the discrepancy rather than guessing which is right; re-run `eval.toolcall_harness` against `config/dev.toml` to resolve. *(not like-for-like — two different documents disagree)* | 0/20 malformed, 20/20 correct decisions, 10/20 parsed (ternary.md) | 0/20 malformed (0.00%), 19/20 correct decisions, 11/20 parsed (granite.md) |
| Single-turn (18 Q) cited / supported / dump / taught | 0.11 / 0.00 / 0.00 / 0.28 (citation_experiment.md RUN A, ternary.md) | 0.17 / 0.00 / 0.00 / 0.11 (ternary_citations.md, ternary.md) | 0.44 / 0.11 / 0.06 / 0.22 (granite_citations.md, granite.md) |
| Helium x3 exact figures + cited | Drifting figures (-268.9/-268.92/-268.9 C), 0/3 cited, 0/3 calc (citation_experiment.md RUN B) | Exact figures 3/3, 1/3 cited, 0/3 calc (ternary.md, ternary_helium.md) | Exact figures 3/3, 1/3 cited, 0/3 calc (granite.md, granite_helium.md) |
| Two-turn cache reuse | Works, no numeric figures on record for this specific check (ternary.md: "works (prior model)") *(not like-for-like — no numbers)* | Works: turn1 prompt_tokens=628, turn2 cached_tokens=1071 (1.7x, over the 0.7x threshold) (ternary.md) | Works: turn1 prompt_tokens=664, turn2 cached_tokens=1219 (Mamba-hybrid prefix reuse confirmed) (granite.md) |
| 10-min soak: turns / errors | 18 / 0 (this run, q1_0_soak10.md) | 29 / 0 (ternary_soak10.md) | 62 / 0 (this run, granite_soak10_v2.md) |
| 10-min soak: first-token p50 / p95 | 25.132s / 56.855s (q1_0_soak10.md) | 8.532s / 74.803s (ternary_soak10.md) | 2.969s / 11.212s (granite_soak10_v2.md) |
| 10-min soak: total p50 | 25.132s (q1_0_soak10.md) | 8.532s (ternary_soak10.md) | 2.977s (granite_soak10_v2.md) |
| 10-min soak: cited_rate | 0.000 (q1_0_soak10.md, new metric) | 0.952 (= 1 - 0.048 uncited; ternary_soak10.md used the **old presence-only metric**, `supported` was not measured for Ternary) | 1.000 (granite_soak10_v2.md, new metric) |
| 10-min soak: supported_rate | 0.000 (q1_0_soak10.md, new metric) | **unknown** — Ternary's soak predates the `supported` metric and was not re-run *(not like-for-like)* | 0.000 (granite_soak10_v2.md, new metric — striking: 100% cited, 0% supported, see by-eye notes below) |
| 10-min soak: calc correctness | 0/0, n/a — no calc items scripted in this run's turns (q1_0_soak10.md) | 3/3 = 1.000 (ternary_soak10.md) | 6/8 = 0.750 (granite_soak10_v2.md) |

## By-eye read of `supported`: is the mechanical flag right?

Read from `data/q1_0_soak10.turns.json` (Bonsai Q1_0, this run) and
`data/granite_soak10_v2.turns.json` (Granite, this run); Ternary was not
re-run so no `.turns.json` exists for it under the new metric.

**Bonsai Q1_0** — trivial case: every factual turn sampled (5/5 read) has
no `[S#]` label anywhere in the answer at all (e.g. "What is
photosynthesis?" -> a fluent multi-paragraph answer citing nothing). The
mechanical `uncited=True` flag is unambiguously correct here; there is no
judgment call to make since there is nothing to check.

**Granite** — the interesting case. Every factual turn sampled (6/6 read)
carries exactly one `[S1]` label, and every one of those labels is flagged
`unsupported`. By eye:
- **Bad example**: "What is photosynthesis?" -> a 5-sentence paragraph
  (light energy conversion, glucose storage, byproduct oxygen, importance
  for life on Earth) with a single `[S1]` tag stapled onto the very last
  word of the whole answer. The mechanical check compares only the final
  citing sentence ("This process is essential for life on Earth...")
  against S1's passage text, and that sentence alone doesn't share enough
  content terms with the passage — so `unsupported` looks correct as a
  read of *that sentence*, but it likely under-credits the paragraph as a
  whole, since earlier sentences may well be grounded in S1's passage even
  though the label isn't attached to them.
- **Consistent pattern, not a one-off**: "Why do plants need sunlight for
  that?", "What gas do they release?", "Where in the cell does it
  happen?", "What's chlorophyll for?" all show the same shape — a single
  trailing `[S1]` after a multi-sentence answer, always flagged
  unsupported.
- **Verdict**: the flag looks **about right, arguably slightly lenient
  toward the model's citation behavior, not too strict** — Granite is
  citing exactly once per turn regardless of how many claims it makes,
  which is a real citation-discipline problem (a single end-of-paragraph
  tag doesn't tell a reader which specific claim is backed by evidence),
  and the mechanical check is correctly detecting that the specific
  sentence carrying the label isn't itself grounded. The check's blind
  spot is that it can't credit a well-placed citation attached to a later
  sentence that summarizes earlier grounded content, so on genuinely
  well-cited answers elsewhere in the corpus (see `granite_citations.md`'s
  sw04 water-cycle example, which *did* clear `supported` at the
  standalone-eval sample rate) it can still work correctly — it just
  never gets a chance to credit anything in this soak's photosynthesis
  thread because Granite's citation habit here is "one tag at the end,"
  not "one tag per grounded claim."

## App handoff (Granite server + app both running)

- `curl http://127.0.0.1:8420/api/status`: `llm.healthy = true`, `model_path
  = "granite-4.0-h-tiny-Q4_K_M.gguf"`, `archives[simplewiki].state =
  "VALID"`.
- Turn: `POST /api/session/{id}/turn {"text": "What is the boiling point of
  helium in celsius and fahrenheit?"}`, read via SSE to `done`.
- Answer (verbatim, one representative run): "Helium boils at
  **-268.928 C**, which is equivalent to **-452.070 F**." True values:
  -268.93 C / -452.07 F — the app's figures are correct to within 0.002 C /
  0.001 F (a much closer, more consistent match than Q1_0's earlier
  drifting figures).
- `citation_quality: "uncited"`, `calc_calls: 0` (no `[S#]` label, and the
  C-to-F conversion was not routed through the `calc` tool despite the
  system prompt's instruction to always use `calc` for unit conversions —
  same model-limited pattern noted in `citation_experiment.md`'s RUN B and
  `granite.md`'s candid assessment).
- Timing (measured with a small urllib script timing the first
  `event: token` SSE line, `data/time_turn.py`, prompt cache warm from
  repeated identical/near-identical questions in this session): time to
  first token **0.159 s**, total turn wall time **0.599 s**. This is much
  faster than the soak's cold/growing-context p50 (2.969 s) because the
  prompt cache was hot for this specific short, repeated question; it is
  not a like-for-like figure against the soak's latency numbers above.
- Both the Granite `llama-server` (port 8080) and the app
  (`tutor.app.main`, port 8420) were left running at the end of this work,
  as instructed.

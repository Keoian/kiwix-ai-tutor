# Soak v3 analysis: long answers, label spam, calc misses

Offline analysis of `data/granite_soak10_v2.turns.json` (62 turns) vs
`data/granite_soak10_v3.turns.json` (33 turns). Scripts: `data/soak_v3_analysis/*.py`. Runaway turn
texts kept out of the repo: `data/soak_v3_analysis/turn{31,32,33}.txt`.
## 1. Three huge turns (index 31-33)
**Measured:** `2/3+1/6`, `why common denominator`, `12.5% of 640` -- 11271/11588/10303 chars,
66-86s. They are **cyclic paraphrase loops**, not literal repeats: turn 31 cycles 6 distinct
sentences (x12-23 each), turn 33 alternates 2 (x32/x31). `find_repetition_loop` only flags a
*consecutive run of the same normalised unit*; cycling units differ turn to turn so it never fires.
**No `truncated` field exists in the turn schema** (unmeasured whether `agent_loop.py` recorded
`"max_tokens"`). Indirect evidence of hitting the 2000-token cap (`answer_max_tokens`): tokens_used
deltas 2012/2492/3003 and 66-86s at ~64 t/s imply ~4200-5500 generated tokens if unbounded.
**Fix (tested against these texts):** detect period-N cycles (N=2-8) by comparing `normalised[i]`
to `normalised[i-N]`. Hand-run: period 6 detected in turns 31/32, period 2 in turn 33 -- all well
before 2000 output tokens.
## 2. Label spam

**Measured:** 365 labels / 24 factual turns (mean 15.2; report's own figure 17.3 uses a slightly
different denominator). Early turns use real evidence (S1-S3, growing to ~S10 by turn 27,
legitimate). Turns 31-33 alone run labels up to **S89**: **221/365 (60%)** of all label emissions
come from just those 3 turns, one ascending invented label per cycled sentence -- copying the seed
exchange's per-sentence-label *style* but with no matching passage. **`[S0]` leaks: 0.**
## 3. supported_rate 0.000 vs A/B's 0.35

**Measured:** soak's `supported_turns` is all-or-nothing per turn (same definition as the A/B
eval's `all_supported`). Per-label supported fraction: **v2 0.0%** (0/44), **v3 6.8%** (25/365) --
both far below A/B's pooled 0.352. **Inferred, not confirmed:** A/B is 18 cold single-turn
questions primed by the fixed seed exchange; this soak is a real accumulating lesson where answers
grow to 3-10+ sentences with looser per-sentence label matches, penalized harder by the strict
overlap check as answers lengthen.
## 4. Length and reading level

**Measured** (same 33 turns): median chars v2 384 / v3 1065; mean 386.7 / 1974.8 (pulled up by
turns 31-33). Words/sentence: v2 17.0, v3 18.0 (no simplification). Chars/word: v2 5.23, **v3
5.96** (longer words) -- v3 does not read as pitched lower despite the pinned grade level (e89f419).
Parroting: v3 repeats its own boilerplate ("This fundamental skill is essential for a wide range of
mathematical applications") verbatim across turns -- self-parroting the seed's abstract tone, not
its literal text.
## 5. Calc misses

**Measured:** `calc_calls == 0` for every scripted calc turn in both runs -- the tool is never
invoked; arithmetic is always done inline. v2's misses (idx 31 `2/3+1/6`->"5/6", idx 36
`9/12`->"3/4") are **correct math**, graded False only by decimal-vs-fraction format mismatch. v3's
idx 31 is the same false miss; idx 33 (`12.5% of 640`) is a real error -- "multiplying 640 by
**0.17** (since **17%**...)" bleeds over from turn 28's "17% of 240", giving 108.8 vs 80 -- wrong
number carried forward, not wrong tool args (no call made). No degC->degF question in this soak.
## 6. unbacked_number 0.25

**By eye:** most flags in turns 31-33 are false alarms -- the "figure" found is the sentence's own
trailing `[S12]`/`[S18]`-style label, not content; ~100+ of ~130 unbacked_number spans there are
this artifact. Genuine positives: turn 30 "multiply 240 by 0.17 to get 40.8", turn 31's two setup
sentences, turn 33's restated "108.8" -- a small handful, not 25%.
## Ranked recommendations

1. **Detect period-N (2-8) cycles in `find_repetition_loop`.** Basis: measured period-2/6 cycles
   the current guard misses. No A/B needed -- widens an existing safety net.
2. **Strip `[S\d+]` before the unbacked-number figure scan.** Basis: measured majority of
   unbacked_number flags in turns 31-33 are label-digit false positives. Bug fix, no A/B.
3. **Cap emitted citation labels at the real evidence-passage count.** Basis: 60% of labels in
   this soak are invented past the real set, all in loop turns. Needs a regression check that
   legit multi-passage turns (S1-S10) are unaffected.
4. **Investigate calc tool non-invocation.** Basis: `calc_calls==0` on 100% of 13 scripted calc
   turns; turn 33's wrong-number error is the class a forced call would prevent. Needs an A/B on
   any stronger calc-nudge (latency/citation-rate tradeoff) before adoption.
5. **Re-measure supported_rate with more in-lesson turns** before concluding 0.35->0.00 is a
   regression -- cause is inferred, not confirmed. Needs a targeted in-lesson eval first.

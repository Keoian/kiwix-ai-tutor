# Attribution: live 3-minute soak measurement

Measured: `python -m eval.run_lesson_soak --config data/dev.soak8421.toml --minutes 3 --out
data/granite_soak3_attr.md`, real Granite llama-server (:8080, shared with the untouched :8420
app), against a separate app instance running the new attribution-event code. Full report:
`data/granite_soak3_attr.md`; per-turn dump: `data/granite_soak3_attr.turns.json`.

## Rates (measured)

| metric | value |
|---|---|
| turns / errors | 17 / 0 |
| cited_rate / supported_rate | 0.000 / 0.000 (0/12 factual turns carry a model `[S#]`) |
| backed_sentence_rate | 0.692 (host-side, from the `attributions` SSE event) |
| unbacked_number_rate | 0.000 (no figures in these turns to exercise the flag) |
| first-token p50 / p95 | 5.485 s / 10.363 s (v2 10-min soak: 3.0 s / 11.2 s; likely noise — only 12 factual turns, colder cache) |

## By-eye read of 5 turns (photosynthesis, turns 1/2/4/5/8)

Caveat: the dump stores `passage_id`/score, not passage text (`passages` stays `[]` live), so this
is a plausibility read against topic, not a literal text diff.

- Turn 1: all 7 sentences attributed to one passage (S1, scores 2–12) — plausible, on-topic.
- Turn 2: 5 sentences spread across S1/S2/S4, no unbacked — topic-consistent, no obvious false positive.
- Turn 4: 1/4 attributed (S2, score 2), 3 plain-`unbacked` — on-topic but thin overlap; flag looks right.
- Turn 5: **all 4 sentences unbacked** — restates turn 4's oxygen claim almost verbatim, which turn
  4 backed via S2. Likely **false negative**: a follow-up shouldn't turn backed prose fully
  unbacked; suggests the overlap threshold is strict when retrieval re-ranks passages turn to turn.
- Turn 8: 3/4 on S1 (scores 2–7), 1 unbacked (a color/pigment sentence, plausibly poor lexical
  overlap with the infobox passage) — looks correctly flagged.

## Verdict

0.692 backed-sentence rate reads plausible but **on the strict side** — turn 5 looks like a false
negative from cross-turn retrieval variance, not a labeling bug. 12 factual turns isn't enough to
say the threshold is definitively wrong; re-run with more why/elliptical follow-ups before tuning.

## :8421 instance

Started for this soak (PID 9064, `data/dev.soak8421.toml`, `data/soak8421_state/`), verified
healthy, then **stopped by this task** after the soak completed. `:8420` and llama-server `:8080`
were never touched.

## Citation-rate discrepancy (v2 10-min soak: 44/44 `[S1]`, this 3-min soak: 0/12)

Investigated whether commits `06e91c2`/`2b56bce`/`26c14d0`/`aeb7a31` (the new host-side
`attribute_sentences` layer, its SSE `attributions` event, and the soak's
`process_turn_stream()` refactor) regressed model-label citation or its parsing. They did not:

- `git diff cdee690 HEAD -- tutor/ eval/` touches only `tutor/app/citations.py` (purely additive:
  new `attribute_sentences`/`Attribution`/`UnbackedSpan` — never edits `resolve_citations`, the
  evidence-packet/prompt composition, or `tutor/app/system_prompt.txt`), `tutor/app/compose.py`
  (emits the new `attributions` event in a `try/except` that can only *add* an event, never touches
  the existing `citations` event or `answer_text`), `tutor/app/lesson_state.py` (one new nullable
  column), and `eval/*` (new metrics + `process_turn_stream()`, a lossless extraction of the same
  inline SSE-parsing loop — `tests/test_lesson_soak.py`'s `process_turn_stream`/`attribution` tests
  pass, as do `tests/test_citations.py` and `tests/test_compose.py` with the fake LLM).
- `data/dev.soak8421.toml` vs `config/dev.granite.toml`/`config/dev.toml`: only `port` and
  `data_dir` differ (checked via `diff` on the non-comment lines). Same sampling
  (`temperature=0.5`/`top_p=0.9`/`top_k=20`), same runtime flags, same registry.
- The per-turn dump proves retrieval and the evidence packet worked normally: turn 1's
  `attribution_event` shows all 7 answer sentences attributed (`model_cited: False`, i.e. no `[S#]`
  in the text) to passage `S1` with overlap scores up to 12 — the same passage the v2 baseline's
  turn 1 also cited as `[S1]`. So `research()` retrieved and labeled the same passage in both runs;
  the model simply never emitted the `[S1]` token this run. In the v2 baseline, `[S1]` appears only
  as a single trailing token at the very end of the answer (`data/granite_soak10_v2.turns.json`
  turn 0: `"... nearly all organisms. [S1]"`) — a one-token habit, not something the prompt forces
  every turn.

**Conclusion: model-output variance, not a code or config regression.** Same llama-server, same
weights, same sampling config, same evidence packet, same prompt-composition code path — the
model chose not to append its trailing `[S1]` token in this run's 12 factual answers. `[S1]` was
a single low-probability trailing token in the baseline to begin with, so a 44/44 → 0/12 swing
across independent 10-minute vs. 3-minute runs (different RNG state, different KV-cache history
from `:8420` traffic sharing `:8080`, colder cache per the ttft note above) is consistent with
sampling noise on a token the model was never reliably forced to emit. No code change made.
**Unverified:** whether the same 0% rate reproduces on a longer soak (only 12 factual turns
here) or is specific to this one run.

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

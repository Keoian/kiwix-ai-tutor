# Evidence-assessor fix: measured false-strong causes and effect

## 1. False-strong causes (measured: JSONL + direct `assess_evidence` calls)

Reviewed `data/probe_rewrite_ab.before_assessor_fix.jsonl`'s 12 strong+
gold-absent turns. Classified by re-running `assess_evidence` on the
recorded titles/terms:

| cause | count | example |
|---|---|---|
| fused/misspelt term split into generic words, each trivially covered by an unrelated, generically-titled passage | 8 | "squarefoot garden" -> square/foot/garden covered by `Garden`, `Foot`, `Chromatica` |
| generic question-topic word ("garden", "work", "food") covering alone, no rare/topical term required | 3 | coverage crossed 0.5 on filler+generic terms with no topical passage present |
| coverage computed over title+**text only**, no requirement the topic itself appear in a title or as a phrase | 1 (overlaps above) | "Square (disambiguation)" text mentions "square" and "foot" separately |

No case found where a corrected term was silently dropped from the
denominator (cause (a) in the task list) -- `_key_terms` already keeps
every original term regardless of correction status; the real bug was
generic-word coverage plus text-only, word-level (not phrase-level)
matching.

## 2. Fix adopted

`tutor/retrieval/assessment.py`:
- New `corrected_terms` optional param (wired from `ResearchResponse`'s
  own `corrected_terms_out` in `research.py`, and from `agent_loop.py`'s
  merged `corrected_terms` dict on the rewrite round).
- `_topic_phrase()`: the question's main topic is the longest corrected
  phrase (e.g. "square foot") if a correction happened, else the longest
  non-generic key term (curated `_GENERIC_SINGLE_WORDS` list of ~20 words
  measured as the offenders above -- a coarse stand-in for real IDF; see
  "deferred" below).
- `_topic_present()`: requires that phrase to appear in a top-3 passage's
  **title**, or as an **exact phrase** in its text -- word-level splits
  across passages no longer count.
- `strong` now requires coverage >= 0.5 **and** `_topic_present`; `reasons`
  field added recording coverage, topic-phrase check, and verdict.
- Threshold unchanged (0.5); no tuning-split score used (Baseline v13
  rationale unchanged).

**Deferred**: rarity-weighting via `research()`'s own `term_matches` /
`estimated_matches` (task candidate (b)). `ResearchResponse` does not
currently surface per-term match counts to callers; plumbing that through
is a larger, separate change. The curated generic-word list plus the
title/phrase requirement addresses every measured false-strong case
without it; recommended follow-up if new false-strong classes appear that
aren't "generic word covers alone."

## 3. Tuning-split check (42 questions, `--split tuning`, real archive)

`python -m eval.run_retrieval_eval ... --split tuning --out
data/tuning_assessor_after.md`: recall@1 0.595, recall@3 0.690, recall@5
0.762, MRR 0.645 (n=42). `assess_evidence` never touches `research()`'s
passages/status (only the additive `.assessment` attribute set after
packing), so these numbers are unaffected by this change **by
construction** -- confirmed by code path, not a diffed before/after re-run
(skipped to conserve time budget). No separate strong/weak confusion
report was generated for the 42 tuning questions this session (would need
`healthy_terms`/gold-title data not carried by `run_retrieval_eval`);
flagged as remaining work below.

## 4. Probe re-measurement (real archive + live llama-server)

New file `data/probe_rewrite_ab.after_assessor_fix.jsonl` (32 records,
16 questions x 2 arms -- driver now supports `--out`). Old data
untouched, copied to `data/probe_rewrite_ab.before_assessor_fix.jsonl`.

| | BEFORE on | BEFORE off | AFTER on | AFTER off |
|---|---|---|---|---|
| n | 25 | 24 | 16 | 16 |
| rewrite_rate | 0.28 | 0.00 | 0.25 | 0.00 |
| gold_in_evidence | 0.54 | 0.50 | 0.58 | 0.55 |
| false_strong (strong & gold absent) | 11 | 10 | 4 | 4 |
| not_found | 0 | 0 | 2 | 0 |
| mean wall_s | 11.1 | 10.8 | 9.2 | 10.1 |
| rescued (gold absent OFF, present ON) | 0 | -- | 1 | -- |

Note: BEFORE/AFTER n differ because this session only had time to
re-run the subset of probes covered in the AFTER file's 16 ids (not the
full 28); BEFORE numbers are over the full historical set. False-strong
dropped from ~44% of strong-and-checked cases to 25% in both arms --
the assessor now correctly downgrades the fused/generic-word cases to
`weak`, which is why 2 "on" turns now honestly answer "couldn't find it"
(`not_found=2`) instead of silently answering off-topic. Mean wall time
improved slightly (assessment adds negligible cost; no rewrite-cost
regression).

Remaining failures (one-word verdict) on turns still not finding gold
after rewrite: rewrites like "square foot gardening basics" or "garden
the right way" are **good queries**, but retrieval doesn't stem
"garden"/"gardening" together in all cases -- verdict: **retrieval
recall** (no stemming across verb/noun forms), not query quality, merge,
or assessment. This matches `docs/rewrite_on_weak_evidence.md`'s own
residual note.

## Recommendation

1. Ship the assessor fix (tests green, tuning recall/MRR unchanged by
   construction, probe false-strong roughly halved).
2. Next: add stemming/lemmatization to lexical retrieval (`garden` <->
   `gardening`) -- the dominant remaining failure class is recall, not
   assessment.
3. Longer term: surface `term_matches`/`estimated_matches` on
   `ResearchResponse` so `assess_evidence` can use real IDF instead of
   the curated generic-word list.

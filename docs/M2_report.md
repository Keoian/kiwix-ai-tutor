# M2 report: lexical retrieval baseline

Gate (`docs/plan/offline_tutor_implementation_plan.md` §1, row **M2**,
"Lexical retrieval baseline"): *"`research()` returns a budgeted packet
with stable IDs/offsets; eval set runs; baseline metrics recorded."*
Work package gate: WP-B6, *"Runs `eval/run_retrieval_eval.py`; baseline
recorded in `docs/retrieval_baseline.md`."*

## Checklist

- [x] **`research()` returns a budgeted packet with stable IDs/offsets.**
  `ResearchEngine.research()` (`tutor/retrieval/research.py`) returns a
  `ResearchResponse` whose `passages` are `ResearchPassage`s carrying a
  stable `passage_id` (fingerprint + path + section + extractor version
  + text hash + span, per WP-B5), exact `text[start:end]` offsets, and
  `estimated_tokens`; packing (`hybrid.packer.pack`) enforces the
  configured token budget (default 2000, mean observed 1091.1 tokens
  over 60 real questions, see `docs/retrieval_baseline.md`).
- [x] **The archive registered.** `simplewiki` tier-1 archive
  (`C:\kiwix\wikipedia_en_simple_all_maxi_2026-05.zim`, 394,566 articles,
  SSD) validated `VALID` with `has_fulltext_index=True`; recorded in
  `docs/archive_registry.md`.
- [x] **Eval set runs against it.** `eval/questions/simplewiki_questions.jsonl`
  (60 questions, 30 tuning / 30 held-out, 7 categories drawn from the
  spec's 60-question structure in §15) run through
  `python -m eval.run_retrieval_eval` against a registry containing only
  `simplewiki`. Every `expected_paths` entry was verified to resolve and
  contain an expected keyword in the real archive before being added
  (see the "Real tier-1 baseline" section of `docs/retrieval_baseline.md`
  for the path-scheme discovery and verification method).
- [x] **Baseline metrics recorded.** `docs/retrieval_baseline.md` now
  has tuning and held-out tables (overall + per-category recall@1/3/5,
  MRR, mean latency), warm-vs-cold latency, `partial` rate (0.0%), mean
  packed tokens (1091.1), and a failure analysis of the 5 most
  instructive misses plus unvalidated next-step hypotheses.

## Evidence summary

| Split | recall@1 | recall@3 | recall@5 | MRR | mean latency (s) |
|---|---|---|---|---|---|
| Tuning (warm, n=30) | 0.467 | 0.500 | 0.500 | 0.481 | 0.652 |
| Tuning (cold, n=30) | 0.467 | 0.500 | 0.500 | 0.481 | 0.922 |
| Held-out (warm, n=30) | 0.567 | 0.600 | 0.600 | 0.578 | 1.126 |

`direct` questions (the category the spec's headline "candidate/
extraction recall >= 95%/90%" target is closest to, single-topic lookup
with no ambiguity) hit recall@1 = 1.000 on both splits -- the pipeline's
lexical core (BM25 + title-suggestion RRF) works as designed for its
intended case. Overall recall is dragged down by categories the current
lexical-only pipeline is not designed to solve yet: `elliptical`
(no antecedent resolution -- out of scope until dense/context-aware
retrieval, WP-B7/M4), and `absent`/`false_premise` (no abstention
signal -- the engine always returns a best-effort lexical guess rather
than `status == "empty"`, the single most consequential gap this eval
surfaced). Full per-category tables and failure analysis are in
`docs/retrieval_baseline.md`.

## Deviations / known gaps

- The spec's full 60-question structure (20 direct / 10 why-how / 10
  comparison / 5 elliptical / 10 ambiguity-false-premise-absent / 5
  tables-formulas-unicode-image) was mapped onto categories meaningful
  for a *retrieval-only* eval (no image-dependent or unicode-rendering
  cases, since those require the LLM/UI layer, not `research()`); the
  10 arithmetic/algebra "student-work-contains-an-error" items are out
  of scope for this milestone (they test the `calc` tool + tutor loop,
  not retrieval).
- `unanswerable`/`absent`/`false_premise` scoring required a small,
  test-first extension to `eval/run_retrieval_eval.py::_score_one`
  (score correct iff `status == "empty"` and no passages returned) plus
  two new unit tests in `tests/test_retrieval_eval.py`; this is a
  measurement-tooling change, not a pipeline change, and does not
  affect the fixture-eval numbers already in `docs/retrieval_baseline.md`.
- No ranking flags were enabled and no parameters were changed this
  milestone; the `title_boost` flag is flagged as a tuning candidate
  for a future WP-B8 eval run, not applied here.
- A dedicated per-query p50/p95 latency percentile script hit a
  Windows file-lock race tearing down the ZIM worker between cold/warm
  phases and was not repeated within the time box; warm/cold means and
  per-category means are recorded instead (see
  `docs/retrieval_baseline.md`).

## Follow-up: Baseline v2 (coverage flags + context, same day)

The two structural gaps this report's failure analysis called out --
no abstention signal, and no conversational context for elliptical
follow-ups -- are fixed; see `docs/retrieval_baseline.md`'s "Baseline
v2: coverage flags + context" section for the full before/after tables,
threshold-tuning notes, and the latency trade-off from gating tier-2
consultation behind coverage (spec §6). Headline held-out recall@5:
overall 0.600 -> 0.700, `elliptical` 0.000 -> 0.667, `absent` 0.000 ->
0.333; `false_premise` and `comparison` are unchanged for reasons that
are ranking/entity-resolution problems, not abstention problems, and
out of scope for a coverage-flag change (ranking flags stay OFF).
`eval/questions/simplewiki_questions.jsonl`'s `false_premise`
`expected_paths` were also corrected from `[]` to the real entity each
question is actually about (verified to exist in the archive), per the
spec's "return evidence about the real entity" framing for false
premises. Mean latency moved further from the <=1 s warm target for the
weak-coverage categories (they now legitimately pay for a tier-2
consultation), which is documented as an open trade-off, not silently
absorbed.

## Test/lint status

`python -m pytest -q -m "not integration"` and `python -m ruff check .`
are green for everything in this milestone's scope
(`eval/`, `tests/test_retrieval_eval.py`, `docs/`); `tests/test_calc_tool.py`
has pre-existing failures caused by concurrent work in `tutor/app/**` by
another agent, out of this milestone's scope and untouched here.

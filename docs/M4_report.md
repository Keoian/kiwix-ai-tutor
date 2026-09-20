# M4 report: dense sidecar + hybrid retrieval

Plan gate (docs/plan/offline_tutor_implementation_plan.md, §1, M4):

> Resumable, fingerprinted embedding pass; hybrid RRF beats lexical baseline
> on tuning without hurting held-out.

All numbers below are from `eval/tools/hybrid_matrix.py` run against the
real Simple Wikipedia archive with the full 281,270-vector bge-small
sidecar, embedding on CPU. The generated tables live in
`docs/hybrid_eval.md`; this report does not alter them.

## Checklist

**1. Resumable, fingerprinted embedding pass** — **PASS**

- Unit tests cover checkpoint/resume and archive-fingerprint staleness
  detection for the dense sidecar (WP-B7/B8; `tutor/retrieval/index/`,
  `tests/test_dense_*`), including the fingerprint mismatch path that
  `tutor/app/compose.py::_open_dense` falls back on (`"stale"` reason).
- The real build completed end-to-end: 281,270 vectors, 216 MB on disk,
  in 3 h 10 min, on the full Simple Wikipedia archive.
- Query performance at full size: p50 17.05 ms, p95 17.85 ms.
- Backfill was verified against all ids in the archive (no missing or
  orphaned vectors).

**2. Hybrid RRF beats lexical baseline on tuning** — **PASS**

Tuning split (n=30), from `docs/hybrid_eval.md`:

| configuration | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| lexical-v2 | 0.533 | 0.567 | 0.567 | 0.548 |
| hybrid | 0.567 | 0.633 | 0.667 | 0.607 |
| hybrid+lead_augmentation (chosen) | 0.567 | 0.667 | 0.733 | 0.628 |

Hybrid, and hybrid+lead_augmentation in particular, beat lexical-v2 on
every tuning metric.

**3. Without hurting held-out** — **FAIL**

Held-out split (n=30, run exactly once):

| configuration | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| lexical-v2 | 0.667 | 0.700 | 0.700 | 0.678 |
| hybrid+lead_augmentation (chosen) | 0.567 | 0.667 | 0.767 | 0.640 |

recall@5 improved (0.700 -> 0.767), but recall@1 (0.667 -> 0.567) and MRR
(0.678 -> 0.640) both got worse. Mean/p95 latency improved substantially
(1.041 -> 0.907 s mean; 2.688 -> 1.797 s p95), which is a genuine win, but
it does not satisfy the gate as written.

**Mechanical gate verdict: FAIL.**

## Interpretation

n=30 per split, so one question is worth 0.033 of any recall metric. The
held-out recall@1 drop is exactly 3 questions (0.100 = 3/30); the recall@5
gain is exactly 2 questions (0.067 = 2/30).

`eval/tools/hybrid_matrix.py`'s `EvalReport` only stores aggregated
recall/MRR per configuration and per category — it does not retain which
individual held-out questions each configuration got right or wrong. A
proper paired sign test (win/loss per question, lexical vs. hybrid) is
therefore **not computable from the existing eval output**, and no such
number is stated below or should be invented.

What can honestly be said with only the aggregates: treating the two
recall@1 rates as independent binomial proportions at n=30 (a conservative
approximation — the true paired test would have a smaller standard error),
the pooled standard error of the difference is about 0.125, versus an
observed difference of 0.100 (z ~= -0.8). For MRR the observed swing
(0.038) is similarly under one standard error of a comparable
approximation. By this measure the held-out comparison is **underpowered
in both directions** — the FAIL is not being overturned by this analysis,
but neither is it being read as a strong, well-powered result. A 3-question
swing at n=30 is well within the noise this sample size can produce.

## What was NOT done, and why

- Held-out was run exactly once, per the plan's rule that a held-out set
  cannot be used, even by accident, to pick among configurations. It is
  not re-run here to check a different flag combination or to try to flip
  the verdict — doing so would invalidate it as held-out evidence.
- Plain `hybrid` (no ranking flags) and `hybrid+title_boost` were never
  scored on held-out at all — only the tuning-chosen configuration
  (`hybrid+lead_augmentation`) and the `lexical-v2` baseline were, per the
  plan's "baseline + chosen only" rule for the held-out run.
- Other notable tuning-only observations, recorded for context: the
  Volcano "elliptical" question is missed in tuning by every
  configuration except at reduced recall (see `docs/hybrid_eval.md`'s
  per-category table, `elliptical` column: lexical-v2/hybrid/etc. all sit
  at or below 0.500, and `hybrid+heading_affinity` drops it to 0.000). The
  coverage gate used elsewhere in retrieval is lexical-only and was not
  exercised by this matrix.

## Recommendations (for the project owner to decide, not decided here)

1. **Enlarge the question set** (e.g. 100+ questions per split) before
   drawing any further conclusion — n=30 cannot reliably resolve a
   3-question swing either way.
2. **Investigate why dense candidates displace the correct top-1**, using
   the TUNING split only (held-out must not be touched again for this
   round of decisions). Two concrete hypotheses worth checking:
   article-level RRF weighting, and `lead_augmentation` promoting lead
   passages of near-neighbour articles ahead of the correct passage.
3. **A fresh held-out set is required for any further gate decision.**
   This held-out set has now been observed (its scores are quoted in this
   report and in `docs/hybrid_eval.md`), so it can no longer serve as
   unbiased held-out evidence for a revised configuration.

## Config status

Per this FAIL, `[embedding].enabled` in `config/dev.toml` is `false` and
`tutor/app/compose.py` only opens the dense sidecar when a config
explicitly sets it `true` (status reason `"disabled in config"`
otherwise). All four ranking flags (`title_boost`, `mention_penalty`,
`heading_affinity`, `lead_augmentation`) remain off by default in
`tutor/retrieval/hybrid/ranking.py::RankingFlags`, per plan §10.8: a flag
is enabled only by a commit that includes its eval table *and* the gate
passing. The gate did not pass, so none of them default on.

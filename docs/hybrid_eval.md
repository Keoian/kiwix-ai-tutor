# Hybrid retrieval matrix (M4 gate)

Chosen configuration: **hybrid+lead_augmentation**

## M4 gate verdict: **FAIL**

## Tuning split

| configuration | recall@1 | recall@3 | recall@5 | mrr | mean latency (s) | p95 latency (s) | dense_used rate | n |
|---|---|---|---|---|---|---|---|---|
| lexical-v2 | 0.533 | 0.567 | 0.567 | 0.548 | 0.941 | 2.484 | 0.000 | 30 |
| hybrid | 0.567 | 0.633 | 0.667 | 0.607 | 0.818 | 2.313 | 0.967 | 30 |
| hybrid+title_boost | 0.567 | 0.700 | 0.700 | 0.627 | 0.713 | 2.266 | 0.967 | 30 |
| hybrid+mention_penalty | 0.567 | 0.667 | 0.700 | 0.612 | 0.739 | 2.203 | 0.967 | 30 |
| hybrid+heading_affinity | 0.500 | 0.600 | 0.633 | 0.550 | 0.745 | 2.203 | 0.967 | 30 |
| hybrid+lead_augmentation | 0.567 | 0.667 | 0.733 | 0.628 | 0.738 | 2.203 | 0.967 | 30 |

## Held-out split (run once, baseline + chosen only)

| configuration | recall@1 | recall@3 | recall@5 | mrr | mean latency (s) | p95 latency (s) | dense_used rate | n |
|---|---|---|---|---|---|---|---|---|
| lexical-v2 | 0.667 | 0.700 | 0.700 | 0.678 | 1.041 | 2.688 | 0.000 | 30 |
| hybrid+lead_augmentation | 0.567 | 0.667 | 0.767 | 0.640 | 0.907 | 1.797 | 1.000 | 30 |

## Per-category recall@5 (tuning)

| configuration | absent | comparison | direct | elliptical | false_premise | tables_formulas | why_how |
|---|---|---|---|---|---|---|---|
| lexical-v2 | 0.000 | 0.400 | 1.000 | 0.500 | 0.333 | 0.333 | 0.400 |
| hybrid | 0.000 | 0.600 | 1.000 | 0.500 | 0.667 | 0.667 | 0.400 |
| hybrid+title_boost | 0.000 | 1.000 | 1.000 | 0.500 | 0.667 | 0.667 | 0.200 |
| hybrid+mention_penalty | 0.000 | 0.800 | 1.000 | 0.500 | 0.667 | 0.667 | 0.400 |
| hybrid+heading_affinity | 0.000 | 0.600 | 1.000 | 0.000 | 0.667 | 0.667 | 0.400 |
| hybrid+lead_augmentation | 0.000 | 1.000 | 1.000 | 0.500 | 0.667 | 0.667 | 0.400 |

## Per-flag verdicts

| flag | verdict | justified by |
|---|---|---|
| title_boost | enable | hybrid+title_boost vs. hybrid, above |
| mention_penalty | enable | hybrid+mention_penalty vs. hybrid, above |
| heading_affinity | keep off | hybrid+heading_affinity vs. hybrid, above |
| lead_augmentation | enable | hybrid+lead_augmentation vs. hybrid, above |

## Interpretation

See `docs/M4_report.md` for the full mechanical-gate checklist, the honest
uncertainty analysis of the held-out numbers above (n=30 per split; the
per-question win/loss breakdown needed for a proper paired test is not
retained by this driver's `EvalReport`, so it is not fabricated here), and
recommendations for the project owner.

The "Per-flag verdicts" table above is a **tuning-only** signal: each
verdict compares that flag against plain `hybrid` on the tuning split
alone. It is not a claim that any flag is safe to enable by default. The
overall **M4 gate verdict is FAIL** (held-out recall@1 and MRR got worse
even though held-out recall@5 improved), and per plan §10.8 a flag is
enabled only by a commit that includes its eval table *and* the gate
passing. Since the gate did not pass, all four flags
(`title_boost`, `mention_penalty`, `heading_affinity`, `lead_augmentation`)
remain off by default regardless of their tuning-only "enable" verdicts
above, and `[embedding].enabled` in `config/dev.toml` defaults to `false`.

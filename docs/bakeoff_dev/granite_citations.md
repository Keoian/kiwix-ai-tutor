| Variant | n | citation_rate | all_resolve_rate | supported_citation_rate | evidence_dump_rate | on_topic_rate | taught_rate | avg_len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 18 | 0.44 | 0.44 | 0.11 | 0.06 | 0.33 | 0.22 | 7660 |

## Manual read (6 answers, sw01-sw06 of the 18)

Baseline for the previous model (Bonsai Q1_0) on these same 18: cited 0.11,
supported 0.00, evidence_dump 0.00, on_topic 0.11, taught 0.28. Granite:
cited 0.44, supported 0.11, evidence_dump 0.06, on_topic 0.33, taught 0.22
— clearly ahead on citation rate and on-topic rate, roughly flat on
"taught", still weak on "supported" (facts actually backed by the cited
sentence).

Read verbatim (see `data/granite_citations.json`, `current[0..5]`):

- **sw01 "What is photosynthesis?"** — no citation. Answer is fluent,
  factually reasonable (equation, chloroplasts, oxygen byproduct), reads at
  roughly a middle-school level, but it is not grounded in the retrieved
  evidence at all — reads like parametric knowledge, not the packet.
- **sw02 "What is the Pythagorean theorem?"** — no citation. Correct and
  clear (a^2+b^2=c^2, Pythagorean triples), but again pure parametric
  knowledge with no [S#] tag; includes a claim ("370 known proofs") that is
  plausible but unverifiable against the evidence shown.
- **sw03 "What is algebra?"** — no citation. Correct, well-explained, uses
  LaTeX-ish notation and touches abstract algebra (groups/rings/fields) that
  is likely above a 10-16-year-old's reading level for this question -
  answer drifts past what a beginner needs.
- **sw04 "What is the water cycle?"** — cited, `[S1]` present, on-topic,
  but NOT flagged "supported" by the scorer: the answer's 6-step numbered
  breakdown (evaporation/condensation/precipitation/collection/
  runoff/transpiration) is more granular than what's likely in a single
  S1 sentence, so some steps are asserted without evidence backing. Reading
  level and structure (bolded numbered list) are good for the target age.
- **sw05 "What is a chemical bond?"** — no citation. Reasonable but denser
  content ("hex-3-ene", "coordination bond") than a 10-16-year-old needs;
  no evidence grounding.
- **sw06 "What is the periodic table?"** — no citation. Correct, decent
  reading level, still no evidence grounding.

No refusals were seen in any of the 18. No odd/broken formatting was seen
(markdown bold/numbered lists render fine; one LaTeX-bracket equation in
sw01 that a plain-text renderer would show ugly, `\[ ... \]`, but that's a
prompt/rendering concern, not a citation defect).

**Label syntax**: checked all 18 answers programmatically for alternate
citation syntaxes (`[1]`, `(S1)`, `(1)`) that our extractor might miss —
**found zero** instances; every citation the model produced used the
correct `[S1]`/`[S2]`... syntax our extractor already parses. So the
`supported_citation_rate` gap here is a genuine "cites but overstates
detail beyond the evidence" issue, not an extractor-syntax miss.

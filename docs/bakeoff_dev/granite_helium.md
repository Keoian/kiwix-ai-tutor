# Helium boiling-point probe (Granite 4.0 H-Tiny, dev.granite.toml), x3

Command: `python -m eval.run_turn_eval --config config/dev.granite.toml
--registry config/archives.simplewiki_only.toml --questions
data/helium_q.jsonl --variants current --out data/granite_helium_run.md`

Question (x3, `data/helium_q.jsonl`): "Output the boiling point of helium in
celsius and farenheit." Evidence S1 (Helium article): "Boiling point: 4.222 K
(−268.928 °C, −452.070 °F)".

## Verbatim answers

- **he01**: "The boiling point of helium is **4.222 K**, which is
  **−268.928 °C** or **−452.070 °F**."
  cites S1? No. Figures match evidence? Yes, exact. Calc called? No.
- **he02**: "The boiling point of helium is **4.222 K**, which is
  **−268.928 °C** or **−452.070 °F**."
  cites S1? No. Figures match evidence? Yes, exact. Calc called? No.
- **he03**: "The boiling point of helium is **4.222 K**, which is
  **−268.928 °C** or **−452.070 °F**.\n\nSources: [S1] Melting point: 0.95 K
  (−272.20 °C, −457.96 °F) (at 2.5 MPa); [S2] Boiling point: 4.222 K
  (−268.928 °C, −452.070 °F)."
  cites S1/S2? Yes (both). Figures match evidence? Yes, exact. Calc called?
  No.

## Summary

- All three figures are byte-for-byte correct against the packet's S1
  (4.222 K / −268.928 °C / −452.070 °F) in all three runs — no hallucinated
  arithmetic, no unit-conversion drift.
- Only 1/3 runs actually emitted a `[S#]` citation (he03); the other two
  gave the same correct numbers with zero citation, i.e. treated it as
  parametric/copied-from-context knowledge without labeling the source.
  This is the same "correct but uncited" pattern seen in the broader
  citation eval (many of the 18 questions above got right answers with no
  `[S#]` tag).
- No calc-tool call in any of the three runs. That's expected/correct here:
  the packet already states the boiling point in all three units, so no
  arithmetic is required; the model correctly copied rather than
  recomputed. (Previous model, Bonsai Q1_0, behaved the same way per
  `data/citation_run_helium.json` — 0/3 calc calls, 0/3 citations, but with
  the added defect of drifting values, e.g. "-268.9°C"/"-452.0°F" instead of
  the precise "-268.928/-452.070" figures Granite reproduced exactly.)
- Flags: none — no refusals, no malformed output, no unsupported claims.
  The only flag is the citation-omission rate (2/3) despite correct facts.

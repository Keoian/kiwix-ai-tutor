# Helium boiling-point probe (Ternary-Bonsai-8B Q2_0_g64, dev.ternary.toml), x3

Command: `python -m eval.run_turn_eval --config config/dev.ternary.toml
--registry config/archives.simplewiki_only.toml --questions
data/helium_q.jsonl --variants current --out data/ternary_helium_run.md`

Question (x3, `data/helium_q.jsonl`): "Output the boiling point of helium in
celsius and farenheit." Evidence S1 (Helium article): "Boiling point: 4.222 K
(−268.928 °C, −452.070 °F)".

## Verbatim answers

- **he01**: "The boiling point of helium is **4.222 K**, which is
  equivalent to **−268.928 °C** or **−452.070 °F**. This information is
  directly supported by [S1]."
  cites S1? Yes. Figures match evidence? Yes, exact. Calc called? No.
- **he02**: "The boiling point of helium is **4.222 K**. To convert this to
  Celsius and Fahrenheit:\n\n- **Celsius**: $4.222\\,K - 273.15 =
  -268.928\\,°C$\n- **Fahrenheit**: $(-268.928\\,°C) \\times 9/5 + 32 =
  -452.070\\,°F$\n\nThus, the boiling point of helium is **-268.928 °C** or
  **-452.070 °F**."
  cites S1? No. Figures match evidence? Yes, exact (arithmetic shown in
  prose happens to land on the same figures as the packet). Calc called?
  No — the model did the conversion itself in prose/LaTeX rather than
  calling the `calc` tool, and rather than just copying the pre-stated
  figures.
- **he03**: "The boiling point of helium is **4.222 K**, which is
  equivalent to **−268.928 °C** or **−452.070 °F**."
  cites S1? No. Figures match evidence? Yes, exact. Calc called? No.

## Summary

- All three runs reproduced the packet's figures byte-for-byte
  (4.222 K / −268.928 °C / −452.070 °F) — no drift, no hallucinated
  arithmetic, no unit-conversion error even in he02 where the model showed
  its own (unnecessary) conversion work instead of copying.
- Only 1/3 runs (he01) emitted an `[S#]` citation; he02 and he03 gave the
  same correct numbers with no citation.
- No calc-tool call in any of the three runs. Two of three (he01, he03)
  correctly copied the pre-stated figures with no computation; he02
  re-derived the conversion in prose instead of calling `calc` or copying —
  it got the right answer but took an unnecessary and slightly riskier path
  (a model that occasionally does this could someday get the arithmetic
  wrong without a tool check).
- Comparison to Granite (docs/bakeoff_dev/granite_helium.md): Granite also
  hit 3/3 exact figures, 1/3 cited, 0/3 calc calls — same pattern.
  Comparison to Bonsai Q1_0 (previous model): that model drifted on
  precision ("-268.9°C"/"-452.0°F" instead of the exact packet figures);
  Ternary (like Granite) reproduces the evidence's precision exactly.
- Flags: none — no refusals, no malformed output, no unsupported claims.
  The only flag is the citation-omission rate (2/3) despite correct facts,
  plus he02's unprompted show-your-work conversion instead of a plain copy
  or a `calc` call.
